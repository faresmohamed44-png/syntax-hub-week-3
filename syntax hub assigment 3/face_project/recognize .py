"""Real-time face detection (OpenCV Haar cascade) & recognition (face_recognition embeddings).

Press q to quit.
"""
import pickle

import cv2
import face_recognition
import numpy as np

TOLERANCE = 0.5  # lower = stricter


def main():
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_encodings, known_names = data["encodings"], data["names"]

    # Detection: OpenCV Haar cascade (ships with opencv-python)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Handles multiple faces: loop over every detected face
        for (x, y, w, h) in faces:
            name = "Unknown"

            # Recognition: embedding for the face found by OpenCV
            # face_recognition expects (top, right, bottom, left)
            box = [(y, x + w, y + h, x)]
            encs = face_recognition.face_encodings(rgb, box)

            if encs and known_encodings:
                distances = face_recognition.face_distance(known_encodings, encs[0])
                best = int(np.argmin(distances))
                if distances[best] <= TOLERANCE:
                    name = known_names[best]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y + h - 30), (x + w, y + h), color, cv2.FILLED)
            cv2.putText(frame, name, (x + 6, y + h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
