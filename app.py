import os
import io
import csv
import threading
import traceback
import json
from datetime import datetime, date, timedelta
from collections import Counter

from flask import (Flask, render_template_string, request, redirect, url_for,
                   send_file, flash, jsonify)
import pandas as pd

# Re-used logic from the provided script (adapted)
import time
import platform
import cv2
import numpy as np
import face_recognition

TRAIN_DIR = "train_images"
CSV_FILE = "Attendance.csv"

# Ensure folders
os.makedirs(TRAIN_DIR, exist_ok=True)

# ---------------- CSV utility functions ----------------

def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "event", "time"])


def write_csv_row(name, event):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, event, now])
        return True, f"{name} -> {event} at {now}"
    except Exception as e:
        return False, str(e)


def read_attendance(last_n=500):
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["name", "event", "time"])
    try:
        df = pd.read_csv(CSV_FILE, engine="python", encoding="utf-8")
        # ensure time column is datetime
        if "time" in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        return df.tail(last_n).iloc[::-1].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["name", "event", "time"])


def export_to_xlsx_bytes():
    df = read_attendance(10000)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf

# ---------------- image loading and encoding ----------------

def load_images_and_names(directory=TRAIN_DIR):
    images = []
    names = []
    if not os.path.isdir(directory):
        return images, names
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        img = cv2.imread(path)
        if img is None:
            continue
        images.append(img)
        names.append(os.path.splitext(file)[0])
    return images, names


def find_encodings(images):
    encode_list = []
    for img in images:
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encs = face_recognition.face_encodings(rgb)
            if face_encs:
                encode_list.append(face_encs[0])
        except Exception:
            continue
    return encode_list

# ---------------- camera handling ----------------

def try_open_camera():
    attempts = []
    system = platform.system().lower()
    if system == "windows":
        attempts.append(lambda: cv2.VideoCapture(0, cv2.CAP_DSHOW))
        attempts.append(lambda: cv2.VideoCapture(1, cv2.CAP_DSHOW))
    else:
        attempts.append(lambda: cv2.VideoCapture(0))
        attempts.append(lambda: cv2.VideoCapture("/dev/video0"))
        try:
            attempts.append(lambda: cv2.VideoCapture(0, cv2.CAP_V4L2))
        except Exception:
            pass

    for attempt in attempts:
        try:
            cap = attempt()
            if cap is None:
                continue
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        except Exception:
            continue
    return None

# ---------------- attendance loop (thread-safe) ----------------

encodings = []
names = []

attendance_thread = None
attendance_stop_event = threading.Event()
attendance_lock = threading.Lock()


def start_encoding_from_images():
    global encodings, names
    images, nms = load_images_and_names()
    encs = find_encodings(images)
    with attendance_lock:
        encodings = encs
        names = nms
    return len(encodings)


def attendance_loop_thread(stop_event):
    # Local copy of encodings/names snapshot
    with attendance_lock:
        encs = list(encodings)
        nms = list(names)

    cap = try_open_camera()
    if cap is None:
        print('[dashboard] No camera available')
        return

    state = {}
    FACE_DISTANCE_THRESHOLD = 0.6  # Strict threshold for face matching (lower is better)
    try:
        while not stop_event.is_set():
            ret, img = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            faces = face_recognition.face_locations(rgb_small)
            encs_found = face_recognition.face_encodings(rgb_small, faces)
            nowt = time.time()

            for encf, faceLoc in zip(encs_found, faces):
                if len(encs) == 0:
                    continue
                faceDis = face_recognition.face_distance(encs, encf)
                idx = np.argmin(faceDis)
                
                # STRICT DISTANCE CHECK: Only mark as match if distance is below threshold
                if faceDis[idx] <= FACE_DISTANCE_THRESHOLD:
                    name = nms[idx].upper()
                    st = state.get(name, {"status": "out", "t0": 0, "cooldown": 0})
                    if st["status"] == "in":
                        if nowt - st["t0"] >= 60:
                            write_csv_row(name, "LOGOUT")
                            st["status"] = "out"
                            st["t0"] = nowt
                            st["cooldown"] = nowt + 60
                            state[name] = st
                    else:
                        if nowt >= st["cooldown"]:
                            write_csv_row(name, "LOGIN")
                            st["status"] = "in"
                            st["t0"] = nowt
                            st["cooldown"] = 0
                            state[name] = st
            # small sleep to reduce CPU
            time.sleep(0.08)
    except Exception:
        traceback.print_exc()
    finally:
        try:
            cap.release()
        except Exception:
            pass

# ---------------- Flask app ----------------

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"

# ---------------- HTML TEMPLATE (enhanced) ----------------

BASE_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Attendance Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" rel="stylesheet">
    <style>
      body { background: linear-gradient(180deg,#f8fafc,#eef2ff); }
      .sidebar { min-height: 100vh; }
      .card-stats { border-radius: 12px; box-shadow: 0 6px 18px rgba(32,33,36,0.08); }
      .small-muted { font-size: 0.85rem; color: #6c757d; }
      .table-wrap { max-height: 420px; overflow:auto; }
    </style>
  </head>
  <body>
  <div class="container-fluid">
    <div class="row">
      <nav class="col-md-2 bg-white sidebar p-3">
        <h4 class="mb-3">Attendance</h4>
        <ul class="nav flex-column">
          <li class="nav-item mb-2"><a class="nav-link" href="#controls">Controls</a></li>
          <li class="nav-item mb-2"><a class="nav-link" href="#summary">Summary</a></li>
          <li class="nav-item mb-2"><a class="nav-link" href="#table">Recent</a></li>
        </ul>
        <hr>
        <div class="small-muted">Encodings: <strong>{{ enc_count }}</strong></div>
        <div class="small-muted">Thread: <strong>{{ thread_running }}</strong></div>
      </nav>

      <main class="col-md-10 p-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h2>Face-Recognition Dashboard</h2>
          <div class="text-end small-muted">Generated: {{ now }}</div>
        </div>

        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="alert alert-info">{{ messages[0] }}</div>
          {% endif %}
        {% endwith %}

        <section id="controls" class="mb-4">
          <div class="card card-stats p-3 mb-3">
            <div class="row g-2">
              <div class="col-md-6">
                <form method="post" action="/export" class="d-inline">
                  <button class="btn btn-primary me-2">Export to XLSX</button>
                </form>
                <form method="post" action="/reencode" class="d-inline">
                  <button class="btn btn-secondary me-2">Recompute Encodings</button>
                </form>
                <form method="post" action="/start" class="d-inline">
                  <button class="btn btn-success me-2">Start Attendance</button>
                </form>
                <form method="post" action="/stop" class="d-inline">
                  <button class="btn btn-danger">Stop Attendance</button>
                </form>
              </div>
              <div class="col-md-6 text-end">
                <form method="post" action="/upload" enctype="multipart/form-data" class="d-inline-block">
                  <input type="file" name="file" class="form-control d-inline-block" style="width:250px;" required>
                  <button class="btn btn-outline-primary mt-2">Upload Image</button>
                </form>
              </div>
            </div>
          </div>
        </section>

        <section id="summary" class="mb-4">
          <div class="row g-3">
            <div class="col-md-3">
              <div class="card p-3 card-stats">
                <div class="small-muted">Total Records</div>
                <div class="h4">{{ stats.total_records }}</div>
                <div class="small-muted">All-time</div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card p-3 card-stats">
                <div class="small-muted">Unique Users</div>
                <div class="h4">{{ stats.unique_users }}</div>
                <div class="small-muted">From training images</div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card p-3 card-stats">
                <div class="small-muted">Logins Today</div>
                <div class="h4">{{ stats.logins_today }}</div>
                <div class="small-muted">{{ stats.today_date }}</div>
              </div>
            </div>
            <div class="col-md-3">
              <div class="card p-3 card-stats">
                <div class="small-muted">Known Encodings</div>
                <div class="h4">{{ enc_count }}</div>
                <div class="small-muted">From train_images/</div>
              </div>
            </div>
          </div>

          <div class="card mt-3 p-3">
            <h6>Attendance (last 30 records)</h6>
            <canvas id="attendanceChart" height="90"></canvas>
          </div>
        </section>

        <section id="table">
          <div class="card p-3">
            <h5>Recent Attendance</h5>
            <div class="table-wrap mt-2">
              <table id="attTable" class="display table table-striped" style="width:100%">
                <thead><tr><th>Name</th><th>Event</th><th>Time</th></tr></thead>
                <tbody>
                {% for _, row in attendance.iterrows() %}
                  <tr><td>{{ row['name'] }}</td><td>{{ row['event'] }}</td><td>{{ row['time'] }}</td></tr>
                {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
        </section>

      </main>
    </div>
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script>
    $(document).ready(function(){
      $('#attTable').DataTable({"pageLength": 8});

      // fetch summary data for chart
      fetch('/api/summary')
        .then(r=>r.json())
        .then(data=>{
          const labels = data.labels;
          const counts = data.counts;
          const ctx = document.getElementById('attendanceChart').getContext('2d');
          new Chart(ctx, {
            type: 'bar',
            data: {
              labels: labels,
              datasets: [{
                label: 'Occurrences',
                data: counts,
                borderRadius: 6,
                barThickness: 22
              }]
            },
            options: {
              responsive: true,
              plugins: { legend: { display: false } },
              scales: { y: { beginAtZero: true } }
            }
          });
        })
        .catch(e=>console.error('Chart load error', e));
    });
  </script>
  </body>
</html>
"""


# ---------------- Flask routes ----------------

@app.route("/", methods=["GET"]) 
def index():
    df = read_attendance(500)
    with attendance_lock:
        cnt = len(encodings)
    running = attendance_thread is not None and attendance_thread.is_alive()

    # compute simple stats
    stats = {
        'total_records': int(len(df)),
        'unique_users': int(len(set(names))) if names else 0,
        'logins_today': 0,
        'today_date': date.today().isoformat()
    }
    if not df.empty and 'time' in df.columns:
        try:
            today = pd.to_datetime(date.today())
            logins_today = df[(df['time'].dt.date == date.today()) & (df['event'].str.upper()=='LOGIN')]
            stats['logins_today'] = int(len(logins_today))
        except Exception:
            stats['logins_today'] = 0

    # format time column for display
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str)

    return render_template_string(BASE_HTML, attendance=df, enc_count=cnt, thread_running=running, now=datetime.now(), stats=stats)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    df = read_attendance(200)
    # create labels & counts (top names in last N records)
    labels = []
    counts = []
    if not df.empty:
        names_list = df['name'].astype(str).tolist()
        c = Counter(names_list)
        top = c.most_common(10)
        labels = [x[0] for x in top]
        counts = [x[1] for x in top]
    return jsonify({ 'labels': labels, 'counts': counts })


@app.route("/export", methods=["POST"])
def export_route():
    try:
        buf = export_to_xlsx_bytes()
        return send_file(buf, as_attachment=True, download_name="Attendance.xlsx", mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        flash(f"Export failed: {e}")
        return redirect(url_for('index'))


@app.route("/reencode", methods=["POST"])
def reencode_route():
    try:
        cnt = start_encoding_from_images()
        flash(f"Recomputed encodings: {cnt}")
    except Exception as e:
        flash(f"Re-encoding failed: {e}")
    return redirect(url_for('index'))


@app.route("/upload", methods=["POST"])
def upload_route():
    file = request.files.get('file')
    if not file:
        flash('No file uploaded')
        return redirect(url_for('index'))
    filename = file.filename
    safe_name = os.path.basename(filename)
    save_path = os.path.join(TRAIN_DIR, safe_name)
    file.save(save_path)
    flash(f'Saved {safe_name} to {TRAIN_DIR}. Click Recompute Encodings.')
    return redirect(url_for('index'))


@app.route("/start", methods=["POST"])
def start_route():
    global attendance_thread, attendance_stop_event
    if attendance_thread is not None and attendance_thread.is_alive():
        flash('Attendance thread already running')
        return redirect(url_for('index'))
    attendance_stop_event.clear()
    attendance_thread = threading.Thread(target=attendance_loop_thread, args=(attendance_stop_event,), daemon=True)
    attendance_thread.start()
    flash('Attendance started')
    return redirect(url_for('index'))


@app.route("/stop", methods=["POST"])
def stop_route():
    global attendance_thread, attendance_stop_event
    if attendance_thread is None or not attendance_thread.is_alive():
        flash('No running attendance thread')
        return redirect(url_for('index'))
    attendance_stop_event.set()
    attendance_thread.join(timeout=5)
    flash('Attendance stopped')
    return redirect(url_for('index'))


if __name__ == '__main__':
    ensure_csv()
    # pre-load encodings (if images present)
    start_encoding_from_images()
    app.run(host='0.0.0.0', port=5000, debug=True)


