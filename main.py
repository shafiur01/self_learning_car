import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Find the model beside this script, even when run from another folder.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def draw_hand(frame, landmarks):
    """Convert hand coordinates to pixels, then draw the skeleton."""
    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


def main():
    if not MODEL_PATH.is_file():
        print(f"Hand model is missing: {MODEL_PATH}")
        return 1

    # Process one frame at a time and track the hand between frames.
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )

    # 'with' closes the detector automatically when this block ends.
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        camera = cv2.VideoCapture(0, backend)

        try:
            if not camera.isOpened():
                print("Could not open the camera.")
                if sys.platform == "darwin":
                    print(
                        "If a camera permission prompt appeared, click Allow, then run again.\n"
                        "Otherwise, open System Settings > Privacy & Security > Camera.\n"
                        "Enable access for the app running Python (such as VS Code or Terminal),\n"
                        "then quit and reopen that app before trying again."
                    )
                print("Also check that a camera is connected and close other camera apps.")
                return 1

            print("Show one hand to the camera. Click the video window and press q to quit.")
            last_timestamp_ms = -1

            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a camera frame.")
                    return 1

                # Mirror the image so moving your hand feels like looking in a mirror.
                frame = cv2.flip(frame, 1)

                # OpenCV uses BGR colors; MediaPipe expects RGB colors.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # VIDEO mode requires a strictly increasing time in milliseconds.
                timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(image, timestamp_ms)

                status = "No hand detected"
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks)
                    status = "Hand detected - 21 landmarks"

                cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                cv2.imshow("Gesture Car - Hand Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
    return 0


# Run the application only when this file is executed directly.
if __name__ == "__main__":
    sys.exit(main())
