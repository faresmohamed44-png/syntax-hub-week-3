"""Capture and register a new person's face images.

Usage: python add_person.py "Ahmed"
Keys:  SPACE = save a photo | q = quit
"""
import os
import sys

import cv2

KNOWN_DIR = "known_faces"


def main():
    if len(sys.argv) < 2:
        print('Usage: python add_person.py "Name"')
        sys.exit(1)

    name = sys.argv[1]
    person_dir = os.path.join(KNOWN_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = len(os.listdir(person_dir))
    print("Press SPACE to capture, q to quit. Take 5-10 photos from different angles.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, f"{name}: {count} photos", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Add person", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            print("Saved", path)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
