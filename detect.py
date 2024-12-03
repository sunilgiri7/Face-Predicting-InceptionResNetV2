import cv2
import numpy as np
import mtcnn
import os
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
from datetime import datetime, timedelta

# Constants
confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
attendance_file = "employee_attendance.xlsx"
temporary_attendance = {}  # Temporary record for the current day
reset_time = datetime.now() + timedelta(days=1)  # Reset attendance daily


# Helper Functions
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def initialize_excel():
    if not os.path.exists(attendance_file):
        # Create a new Excel file with necessary headers if it doesn't exist
        df = pd.DataFrame(columns=["Name", "IN_Time", "OUT_Time", "Date"])
        df.to_excel(attendance_file, index=False)


def log_attendance(name):
    global reset_time
    current_time = datetime.now()
    current_date = current_time.date()

    # Reset temporary attendance if a new day starts
    if current_time >= reset_time:
        temporary_attendance.clear()
        reset_time = current_time + timedelta(days=1)

    # Check if the person already has an entry for today
    if name not in temporary_attendance:
        temporary_attendance[name] = {"IN": None, "OUT": None}

    # Load existing data
    df = pd.read_excel(attendance_file)

    # Mark "IN" or "OUT"
    if temporary_attendance[name]["IN"] is None:
        # First entry (IN time)
        temporary_attendance[name]["IN"] = current_time
        new_entry = {
            "Name": name,
            "IN_Time": current_time.strftime("%H:%M:%S"),
            "OUT_Time": None,
            "Date": current_date,
        }
        df = df.append(new_entry, ignore_index=True)
        print(f"IN entry logged for {name} at {current_time.strftime('%H:%M:%S')}")
    elif temporary_attendance[name]["OUT"] is None:
        # Second entry (OUT time)
        temporary_attendance[name]["OUT"] = current_time
        idx = df[(df["Name"] == name) & (df["Date"] == current_date)].index
        if not idx.empty:
            df.loc[idx[0], "OUT_Time"] = current_time.strftime("%H:%M:%S")
        print(f"OUT entry logged for {name} at {current_time.strftime('%H:%M:%S')}")

    # Save updated data
    df.to_excel(attendance_file, index=False)


def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'
        is_marked = False

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            # Log attendance
            is_marked = True
            log_attendance(name)

            # Draw bounding box and label
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)

            # Add "Marked" tag at the bottom center of the bounding box
            bottom_center = ((pt_1[0] + pt_2[0]) // 2, pt_2[1] + 20)
            cv2.putText(img, "Marked", bottom_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img


if __name__ == "__main__":
    initialize_excel()
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
