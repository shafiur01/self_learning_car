"""Record labeled motion clips or try a trained model. Outputs are screen-only."""

import argparse
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np

from .data import DATA, KEYS, LABELS, SECONDS, CommandGate, features, quality, save_clip


def snapshot(buffer):
    return (np.stack([item[1] for item in buffer]),
            np.array([item[0] for item in buffer]),
            np.array([item[2] for item in buffer], dtype=bool))


def probabilities(model, xyz, times, visible, aspect):
    import torch
    x = features(xyz, times, visible, aspect)
    with torch.inference_mode():
        return model(torch.from_numpy(x)[None]).softmax(dim=1)[0].numpy()


def overlay(cv2, frame, lines):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28 * len(lines) + 10), (15, 15, 15), -1)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (10, 25 + index * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (255, 255, 255), 1, cv2.LINE_AA)


def main(profile=None, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Saved model.pt; omit to collect initial examples")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(argv)
    # Research profiles supply a separate taxonomy/storage/model loader while
    # retaining this same camera, tracking, replay, and labeling workflow.
    data_folder = DATA if profile is None else profile.data
    keys = KEYS if profile is None else profile.keys
    labels = LABELS if profile is None else profile.labels
    save_recording = save_clip if profile is None else profile.save_clip
    new_gate = CommandGate if profile is None else profile.gate
    extra_metadata = {} if profile is None else profile.metadata
    if args.model and args.split != "train":
        parser.error("Live corrections are training data. Record held-out sessions without --model.")

    # Import the existing tracker without starting its camera loop.
    import cv2
    import mediapipe as mp
    from main import MODEL_PATH, draw_hand

    if not MODEL_PATH.is_file():
        parser.error("Missing hand model; see models/README.md.")
    model = None
    if args.model:
        from .model import load_model
        try:
            model, _ = (load_model if profile is None else profile.load_model)(args.model)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))

    session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    print(f"Session: {session}; split: {args.split}")
    print("Directions refer to the mirrored preview. No motor connection is enabled.")
    print("Perform a gesture for 2 seconds, press SPACE, review replay, then label:")
    print(" | ".join(f"{key} {label}" for key, label in keys.items()))
    print("x discards a frozen clip; q exits. Each save requires a fresh recording.")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        # Detect two so ambiguous scenes are rejected instead of switching people/hands.
        num_hands=2,
    )
    gate = new_gate()
    buffer = deque()
    frozen = None
    last_timestamp = -1
    last_prediction = 0.0
    previous_identity = None
    previous_visible = False
    predicted, command = "COLLECTING", "STOP"
    score = 0.0
    message = "Show one hand; SPACE freezes the last 2 seconds."
    window = "Gesture Commands - Screen Demo" if profile is None else profile.window
    hints = [f"{key} {label.lower().replace('_', ' ')}" for key, label in keys.items()]

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        camera = cv2.VideoCapture(args.camera, backend)
        try:
            if not camera.isOpened():
                parser.exit(1, "Camera unavailable. Close the previous camera demo and check camera permissions.\n")
            while True:
                if frozen is not None:
                    # Replay the exact images that belong to the saved landmark sequence.
                    elapsed = (time.monotonic() - frozen["started"]) % frozen["duration"]
                    index = np.searchsorted(frozen["times"] - frozen["times"][0], elapsed, side="right") - 1
                    view = frozen["views"][max(0, index)].copy()
                    overlay(cv2, view, ["FROZEN REPLAY - output STOP", frozen["prediction_text"],
                                      "Label: " + " | ".join(hints[:4]),
                                      " | ".join(hints[4:]), "x discard | q quit"])
                    cv2.imshow(window, view)
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord("q"):
                        break
                    if key in [ord(k) for k in keys]:
                        label = keys[chr(key)]
                        if label != "NO_COMMAND" and not quality(frozen["times"], frozen["visible"]):
                            print("Clip has insufficient tracking. Choose n for no command, or x and record again.")
                            continue
                        metadata = {
                            **extra_metadata,
                            "label": label, "split": args.split, "session": session,
                            "aspect": frozen["aspect"], "source": "correction" if model else "recording",
                            "model": str(args.model.resolve()) if model else None,
                            "prediction": frozen["prediction"], "scores": frozen["scores"],
                            "created_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        path = save_recording(data_folder, frozen["xyz"], frozen["times"], frozen["visible"], metadata)
                        print(f"Saved {label}: {path.name}")
                        message = f"Saved {label}. Perform a NEW gesture, then SPACE."
                    elif key == ord("x"):
                        message = "Discarded. Perform a new gesture, then SPACE."
                    else:
                        continue
                    frozen = None
                    buffer.clear()
                    gate.reset()
                    predicted, command, score = "COLLECTING", "STOP", 0.0
                    previous_identity, previous_visible = None, False
                    continue

                success, frame = camera.read()
                if not success:
                    print("Camera frame unavailable. Output STOP.")
                    return 1
                # Inference uses the original orientation; only the preview is mirrored.
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (640, round(height * 640 / width)))
                height, width = frame.shape[:2]
                aspect = width / height
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
                last_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                now = time.monotonic()
                visible = len(result.hand_landmarks) == 1
                identity = result.handedness[0][0].category_name if visible else None
                # Never blend hand loss, long stalls, or a detected hand switch into old motion.
                if (visible != previous_visible or
                    (visible and identity != previous_identity) or
                    (buffer and now - buffer[-1][0] > 0.3)):
                    buffer.clear()
                    gate.reset()
                    predicted, command, score = "COLLECTING", "STOP", 0.0
                previous_identity, previous_visible = identity, visible
                xyz = np.zeros((21, 3), dtype=np.float32)
                if visible:
                    landmarks = result.hand_landmarks[0]
                    xyz = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
                    draw_hand(frame, landmarks)
                view = cv2.flip(frame, 1)
                buffer.append((now, xyz, visible, view.copy()))
                while len(buffer) > 1 and buffer[0][0] < now - SECONDS:
                    buffer.popleft()
                xyz_clip, times, visibility = snapshot(buffer)
                ready = quality(times, visibility)
                if not ready:
                    gate.reset()
                    command, score = "STOP", 0.0
                    predicted = "COLLECTING" if visible else "NO_HAND / MULTIPLE_HANDS"
                elif model is not None and now - last_prediction >= 0.2:
                    scores = probabilities(model, xyz_clip, times, visibility, aspect)
                    predicted, command = gate.update(scores, tracked=True)
                    score = float(scores.max())
                    last_prediction = now
                status = f"Prediction: {predicted} ({score:.0%} score) | Output: {command}" if model else "Recording mode - no trained model loaded"
                overlay(cv2, view, [status, f"Session split: {args.split} | buffer: {times[-1]-times[0]:.1f}s",
                                  message, "SPACE: freeze/review | q: quit | screen only"])
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if len(times) < 12 or times[-1] - times[0] < SECONDS * 0.85:
                        message = "Wait for about 2 seconds of uninterrupted capture."
                        continue
                    scores = probabilities(model, xyz_clip, times, visibility, aspect) if model and ready else None
                    prediction = labels[int(scores.argmax())] if scores is not None else None
                    frozen = {"xyz": xyz_clip.copy(), "times": times.copy(), "visible": visibility.copy(),
                              "aspect": aspect, "views": [item[3] for item in buffer],
                              "started": time.monotonic(), "duration": times[-1]-times[0],
                              "prediction": prediction, "scores": scores.tolist() if scores is not None else None,
                              "prediction_text": f"Raw model guess: {prediction}" if prediction else "Recording - choose the true label"}
                    gate.reset()
                    command = "STOP"
        finally:
            camera.release()
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
