# self_learning_car

First Learning milestone: We display a webcam feed with 21 tracked hand landmarks.
We used OpenCV for the camera and MediaPipe's pretrained Hand Landmarker for detection.
It does not yet classify gestures or control motors. Frames are processed locally;
this program does not record or upload them anywhere yet. It's just a bare minimum initial progress of computer vision model for the carbot(our main product)

## For Running on macOS (these needs to be followed)

Test dependency setup: Python 3.14 on Apple silicon.
MediaPipe is pinned to 0.10.32 because 1.0.1 crashed during model initialization
on this Mac. Model loading and blank-frame inference passed with 0.10.32.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the model as described in [models/README.md](models/README.md), then run:

```bash
python main.py
```

Allow camera access for the app running Python. Show one hand to see red points
and green connections; remove it to see "No hand detected". Click the video
window and press lowercase **q** to exit.

If the camera fails to open, check System Settings > Privacy & Security > Camera,
enable the app running Python, and quit/reopen that app.

MediaPipe requires `opencv-contrib-python`, which supplies `cv2`. Use that package
alone rather than also installing `opencv-python`, because both own the same files.
