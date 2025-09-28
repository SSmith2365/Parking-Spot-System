import cv2
import os
import time
import numpy as np

from datetime import datetime
from pathlib import Path

# Stream URL (placeholder for GitHub safe code)
stream_url = "rtsp://<username>:<password>@<ip>:<port>/stream1"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

ROOT = Path(__file__).resolve().parents[1]
snapshot_dir = ROOT / "data" / "snapshot"
snapshot_dir.mkdir(parents=True, exist_ok=True)

cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

# Freeze detection setup
prev_frame = None
freeze_count = 0
freeze_seconds = 10  
fps_estimate = 30
freeze_frame_threshold = freeze_seconds * fps_estimate

# Snapshot timer setup
last_snapshot_time = time.time()
snapshot_interval = 300

try:
    while True:
        ret, frame = cap.read()

        if ret:
            # Freeze detection
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame)
                if not np.any(diff):
                    freeze_count += 1
                    print(f"Warning: Stream may be frozen ({freeze_count}/{freeze_frame_threshold})")
                else:
                    freeze_count = 0
            prev_frame = frame.copy()

            if freeze_count >= freeze_frame_threshold:
                print("Stream frozen. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(stream_url)
                freeze_count = 0
                continue  

            # Show live feed
            cv2.imshow("Live Stream", frame)

            # Save snapshot every 30 seconds
            if time.time() - last_snapshot_time >= snapshot_interval:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(snapshot_dir, f"frame_{timestamp}.jpg")
                # filename = f"snapshot/frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved snapshot: {filename}")
                last_snapshot_time = time.time()

            # Quit key
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break

        else:
            print("Error: Failed to capture frame from stream.")
            cap.release()
            time.sleep(5)
            cap = cv2.VideoCapture(stream_url)

except KeyboardInterrupt:
    print("Interrupted by user.")

cap.release()
cv2.destroyAllWindows()

