"""Build face embeddings for every person in known_faces/ and save to encodings.pkl."""
import os
import pickle

import face_recognition

KNOWN_DIR = "known_faces"
OUT_FILE = "encodings.pkl"


def main():
    encodings, names = [], []

    for person in sorted(os.listdir(KNOWN_DIR)):
        person_dir = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for file in os.listdir(person_dir):
            path = os.path.join(person_dir, file)
            image = face_recognition.load_image_file(path)
            faces = face_recognition.face_encodings(image)

            if len(faces) != 1:
                print(f"Skipped {path} (found {len(faces)} faces)")
                continue

            encodings.append(faces[0])
            names.append(person)
            print("Encoded", path)

    with open(OUT_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

    print(f"Saved {len(names)} encodings to {OUT_FILE}")


if __name__ == "__main__":
    main()
