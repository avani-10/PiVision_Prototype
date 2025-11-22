import os
import time
import platform
import csv
from datetime import datetime

import cv2
import numpy as np
import face_recognition
import pandas as pd


TRAIN_DIR = "train_images"
CSV_FILE = "Attendance.csv"


# ------------------ CSV FUNCTIONS ------------------

def ensure_csv():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "event", "time"])
        print("[INFO] Created new Attendance.csv")


def write_csv_row(name, event):
    """Append safely to CSV using csv.writer."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, event, now])
        print(f"[LOG] {name} -> {event} at {now}")
    except Exception as e:
        print("[ERROR] Could not write CSV row:", e)


def view_attendance():
    """Show last 50 attendance entries."""
    if not os.path.exists(CSV_FILE):
        print("[INFO] No attendance file found.")
        return
    print("\n----- Last 50 Entries -----")
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-50:]
        for line in lines:
            print(line.strip())
    print("---------------------------\n")


def export_to_xlsx():
    """Convert CSV → XLSX safely."""
    try:
        df = pd.read_csv(CSV_FILE, engine="python", encoding="utf-8")
        df.to_excel("Attendance.xlsx", index=False)
        print("[OK] Exported to Attendance.xlsx")
    except Exception as e:
        print("[ERROR] Cannot export to XLSX:", e)


# ------------------ IMAGE LOADING ------------------

def load_images(directory):
    images = []
    names = []
    if not os.path.isdir(directory):
        print("[WARN] train_images folder missing.")
        return images, names

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        img = cv2.imread(path)
        if img is None:
            continue
        images.append(img)
        names.append(os.path.splitext(file)[0])
    print("[INFO] Loaded:", names)
    return images, names


def find_encodings(images):
    encode_list = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_enc = face_recognition.face_encodings(rgb)
        if face_enc:
            encode_list.append(face_enc[0])
    return encode_list


# ------------------ CAMERA HANDLING ------------------

def try_open_camera():
    """Try Raspberry Pi cam, USB cam, fallback attempts."""
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
        except:
            pass

    for attempt in attempts:
        cap = attempt()
        if cap is None:
            continue
        ok, _ = cap.read()
        if ok:
            print("[OK] Camera opened successfully.")
            return cap
        cap.release()

    print("[ERROR] No camera found.")
    return None


# ------------------ ATTENDANCE LOOP ------------------

def attendance_loop(encodings, names):
    cap = try_open_camera()
    if cap is None:
        return

    state = {}  # name → {status, t0, cooldown_until}
    FACE_DISTANCE_THRESHOLD = 0.6  # Strict threshold for face matching

    print("[INFO] Press 'q' to stop.\n")

    while True:
        ret, img = cap.read()
        if not ret:
            print("[WARN] Camera read failed.")
            continue

        small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb_small)
        encodes = face_recognition.face_encodings(rgb_small, faces)
        now = time.time()

        for enc, faceLoc in zip(encodes, faces):
            faceDis = face_recognition.face_distance(encodings, enc)

            if len(faceDis) == 0:
                continue

            idx = np.argmin(faceDis)
            
            # STRICT DISTANCE CHECK: Only mark as match if distance is below threshold
            if faceDis[idx] <= FACE_DISTANCE_THRESHOLD:
                name = names[idx].upper()

                # Draw bounding box
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)

                # Auto login/logout logic
                st = state.get(name, {"status": "out", "t0": 0, "cooldown": 0})

                if st["status"] == "in":
                    if now - st["t0"] >= 60:
                        write_csv_row(name, "LOGOUT")
                        st["status"] = "out"
                        st["t0"] = now
                        st["cooldown"] = now + 60
                        state[name] = st

                else:
                    if now >= st["cooldown"]:
                        write_csv_row(name, "LOGIN")
                        st["status"] = "in"
                        st["t0"] = now
                        st["cooldown"] = 0
                        state[name] = st

        cv2.imshow("Attendance Camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------ MAIN MENU ------------------

def main_menu():
    ensure_csv()

    images, names = load_images(TRAIN_DIR)
    encodings = find_encodings(images)
    print(f"[INFO] Encoded {len(encodings)} faces.")

    while True:
        print("\n---- Attendance System ----")
        print("1) Mark Attendance")
        print("2) Export to XLSX")
        print("3) View Attendance")
        print("4) Exit")

        choice = input("Choose [1-4]: ").strip()

        if choice == "1":
            if len(encodings) == 0:
                print("[ERROR] No encodings found. Add images.")
                continue
            attendance_loop(encodings, names)

        elif choice == "2":
            export_to_xlsx()

        elif choice == "3":
            view_attendance()

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid option.")


# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    main_menu()
