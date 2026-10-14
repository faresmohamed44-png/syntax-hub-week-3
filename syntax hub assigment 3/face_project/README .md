# Face Detection & Recognition

Detect and recognize faces in real time using OpenCV and the `face_recognition` library.

## Setup
```
pip install -r requirements.txt
```
On Windows, `face_recognition` needs `dlib`. If install fails, install CMake and Visual Studio Build Tools, or try `pip install dlib-bin`.

## Usage
1. Register a person (SPACE to capture 5-10 photos, q to quit):
   ```
   python add_person.py "Ahmed"
   ```
2. Build embeddings:
   ```
   python encode_faces.py
   ```
3. Run recognition (q to quit):
   ```
   python recognize.py
   ```

## How it works
- OpenCV's Haar cascade (`detectMultiScale`) detects faces in each frame.
- `face_encodings` creates a 128-d embedding per detected face.
- `face_distance` compares against known embeddings; the closest match under the tolerance (0.5) wins, otherwise "Unknown".
- Bounding boxes and name labels are drawn for every face in the frame.

## Structure
```
known_faces/<Name>/*.jpg
add_person.py
encode_faces.py
recognize.py
requirements.txt
```
