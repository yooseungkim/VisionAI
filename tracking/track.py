import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# 1. Configuration
video_name = "parkinglot2.mp4"
video_path = f"datasets/{video_name}"
# Ensure output directory exists
os.makedirs("results", exist_ok=True)
output_path = f"results/processed_{video_name}"

# 2. Color Settings (BGR Format)
# Define colors for specific class IDs (e.g., 0: Person, 2: Car for COCO)
CLASS_COLORS = {
    0: (0, 0, 255),    # Red for Person
    2: (255, 0, 0),    # Blue for Car
    # Add other classes if needed
}
DEFAULT_MASK_COLOR = (0, 255, 0)  # Green for others
ALPHA = 0.5  # Transparency of the mask

# 3. Model & Video Setup
model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture(video_path)

# Get video properties for VideoWriter
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter
# codec 'mp4v' is widely supported for .mp4
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

track_history = defaultdict(lambda: [])

print(f"Processing video: {video_path} -> {output_path}")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Run tracking
    results = model.track(frame, persist=True)
    result = results[0]

    # --- Custom Mask Visualization Start ---
    if result.masks is not None:
        overlay = frame.copy()

        # Extract mask contours (xy) and class IDs
        # result.masks.xy provides list of points [N, 2]
        contours = result.masks.xy
        cls_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []

        for cnt, cls_id in zip(contours, cls_ids):
            # Choose color based on class
            color = CLASS_COLORS.get(cls_id, DEFAULT_MASK_COLOR)
            # Draw filled polygon on overlay
            cv2.fillPoly(overlay, [np.int32(cnt)], color)

        # Blend overlay with original frame
        # Formula: dst = src1*alpha + src2*beta + gamma
        frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)
    # --- Custom Mask Visualization End ---

    # Draw Boxes & Labels (exclude masks since we drew them manually)
    # Passing 'img=frame' draws annotations on top of our masked frame
    annotated_frame = result.plot(img=frame, masks=False)

    # Draw Tracks
    if result.boxes and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))

            if len(track) > 30:
                track.pop(0)

            # Draw tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(
                230, 230, 230), thickness=2)

    # Write frame to video file
    out.write(annotated_frame)

    # Display
    cv2.imshow("YOLO11 Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete.")
print("Processing complete.")
