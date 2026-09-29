# Hand tracking model

`hand_landmarker.task` is Google's pretrained Hand Landmarker bundle. It is
downloaded locally and excluded from Git because it is a binary model asset.

Source: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

From the project directory, download it with:

```bash
curl --fail --location https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -o models/hand_landmarker.task
```
