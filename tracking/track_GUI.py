import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
video_name = "parkinglot1.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)

# 1. Calibration Settings
REAL_WIDTH = 2.5
REAL_HEIGHT = 5.0
DEST_POINTS = np.array([
    [0, 0], [REAL_WIDTH, 0], [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
], dtype=np.float32)

# 2. Filtering Parameters
CONF_THRESHOLD = 0.3   # 신뢰도 낮은 객체 제외
SMOOTH_WINDOW = 5      # 평활화
SPEED_THRESH = 1.0     # 2.0 km/h (낮춰도 됨, 형상 필터가 막아줌)

# [NEW] Shape Stability Parameter
# 이동 거리(Translation)가 크기 변화(Deformation)의 N배보다 커야 함
# 1.0 = 이동 거리가 변형량보다 커야 함 (보수적)
# 0.8 = 이동 거리가 변형량의 80%는 넘어야 함 (약간 여유)
SHAPE_STABILITY_RATIO = 0.4

# State Logic
MAX_SCORE = 300  # 관성 최댓값
START_THRESH = 30
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 60
MIN_TOTAL_MOVE = 0.8
STOP_DECAY_RATE = 0.4

output_path = f"results/processed_shape_filter_{video_name}_STABILITY{SHAPE_STABILITY_RATIO}_START{START_THRESH}"

# --- Mouse Callback ---
mouse_points = []
calibration_done = False


def mouse_callback(event, x, y, flags, param):
    global mouse_points, calibration_done
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < 4:
            mouse_points.append((x, y))
        if len(mouse_points) == 4:
            calibration_done = True

# --- Helper Class ---


class BoxSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update_and_get_avg(self, box):
        self.window.append(box)
        return np.mean(self.window, axis=0)


# --- Main Logic ---
cap = cv2.VideoCapture(video_path)
FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0:
    FPS = 30.0
model = YOLO("yolo11m-seg.pt")

# Calibration
cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_callback)
success, first_frame = cap.read()
if not success:
    exit()
while not calibration_done:
    img = first_frame.copy()
    for pt in mouse_points:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    if len(mouse_points) > 1:
        cv2.polylines(img, [np.array(mouse_points)], False, (0, 255, 255), 2)
    cv2.imshow("Calibration", img)
    if cv2.waitKey(10) == ord('q'):
        exit()
    if len(mouse_points) == 4:
        cv2.waitKey(500)
        break
cv2.destroyWindow("Calibration")
source_points = np.array(mouse_points, dtype=np.float32)
matrix = cv2.getPerspectiveTransform(source_points, DEST_POINTS)

# Data Structures
smoothers = defaultdict(lambda: BoxSmoother(window_size=SMOOTH_WINDOW))
prev_positions = {}   # (rx, ry) in meters
prev_box_dims = {}    # [NEW] (w, h) in pixels for shape check
motion_scores = defaultdict(int)
vehicle_states = defaultdict(lambda: "Stopped")
state_locked_until = defaultdict(int)
start_positions = {}

w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*'avc1'), int(FPS), (w, h))

frame_idx = 0
print(f"Processing... Shape Ratio: {SHAPE_STABILITY_RATIO}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    results = model.track(frame, persist=True, classes=[
                          2, 3, 5, 7], verbose=False)
    result = results[0]

    cv2.polylines(
        frame, [np.array(mouse_points, dtype=np.int32)], True, (100, 100, 100), 2)

    if result.boxes and result.boxes.id is not None:
        raw_boxes = result.boxes.xywh.cpu().numpy()
        raw_ids = result.boxes.id.int().cpu().tolist()
        raw_confs = result.boxes.conf.cpu().tolist()

        # Filter Low Confidence
        boxes, track_ids, yolo_confs = [], [], []
        for b, i, c in zip(raw_boxes, raw_ids, raw_confs):
            if c >= CONF_THRESHOLD:
                boxes.append(b)
                track_ids.append(i)
                yolo_confs.append(c)

        if not boxes:
            out.write(frame)
            cv2.imshow("Shape Filter Tracking", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # Smoothing
        smoothed_boxes = []
        for box, tid in zip(boxes, track_ids):
            smoothed_boxes.append(smoothers[tid].update_and_get_avg(box))

        # Transform
        bottom_centers = []
        for box in smoothed_boxes:
            x, y, w_b, h_b = box
            bottom_centers.append([x, y + h_b / 2])

        pts = np.array(bottom_centers).reshape(-1, 1, 2).astype(np.float32)
        trans_pts = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)

        for (rx, ry), box, tid, conf in zip(trans_pts, smoothed_boxes, track_ids, yolo_confs):
            x, y, w_b, h_b = box  # Smoothed box (pixels)

            # --- 1. Speed Calculation (Meters) ---
            speed_kmh = 0.0
            if tid in prev_positions:
                prev_rx, prev_ry = prev_positions[tid]
                dist_step = math.sqrt((rx - prev_rx)**2 + (ry - prev_ry)**2)
                speed_kmh = dist_step * FPS * 3.6
            prev_positions[tid] = (rx, ry)

            # --- 2. [NEW] Shape Deformation Logic (Pixels) ---
            is_bodily_motion = False  # 통째로 움직이는가?
            deformation = 0.0
            translation = 0.0

            if tid in prev_box_dims:
                prev_w, prev_h, prev_cx, prev_cy = prev_box_dims[tid]

                # Deformation (크기 변화량): 가로 변화 + 세로 변화
                deformation = abs(w_b - prev_w) + abs(h_b - prev_h)

                # Translation (중심점 이동량): 픽셀 단위
                translation = math.sqrt((x - prev_cx)**2 + (y - prev_cy)**2)

                # [KEY LOGIC] 이동량이 변형량보다 커야 함 (비율 필터)
                # 속도가 아주 빠르면(>10km/h) 변형 무시하고 인정 (원근감 변화 때문)
                if speed_kmh > 10.0:
                    is_bodily_motion = True
                elif translation > (deformation * SHAPE_STABILITY_RATIO):
                    is_bodily_motion = True
                else:
                    is_bodily_motion = False  # 제자리에서 찌그러지는 중 (Noise)
            else:
                is_bodily_motion = True  # 첫 프레임은 판단 불가하므로 일단 Pass

            # 다음 프레임 비교를 위해 저장
            prev_box_dims[tid] = (w_b, h_b, x, y)

            # --- 3. Motion Score Update ---
            # 속도 조건 AND 형상 조건 모두 만족해야 함
            if (speed_kmh > SPEED_THRESH) and is_bodily_motion:
                motion_scores[tid] += 1
                if tid not in start_positions:
                    start_positions[tid] = (rx, ry)
            else:
                motion_scores[tid] -= STOP_DECAY_RATE
                if motion_scores[tid] <= 0 and tid in start_positions:
                    del start_positions[tid]

            motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

            # --- 4. State Logic ---
            current_state = vehicle_states[tid]
            is_locked = frame_idx < state_locked_until[tid]
            new_state = current_state

            if not is_locked:
                if current_state == "Stopped":
                    if motion_scores[tid] >= START_THRESH:
                        total_dist = 0
                        if tid in start_positions:
                            sx, sy = start_positions[tid]
                            total_dist = math.sqrt((rx - sx)**2 + (ry - sy)**2)

                        if total_dist >= MIN_TOTAL_MOVE:
                            new_state = "Moving"
                            print(
                                f"[LVLM QUERY] ID {tid} Moving! (Speed:{speed_kmh:.1f}, BodyMove:{is_bodily_motion})")
                            state_locked_until[tid] = frame_idx + \
                                LOCK_DURATION_FRAMES
                else:
                    if motion_scores[tid] <= STOP_THRESH:
                        new_state = "Stopped"
                        state_locked_until[tid] = frame_idx + \
                            LOCK_DURATION_FRAMES

            vehicle_states[tid] = new_state

            # --- Visualization ---
            color = (0, 255, 0) if vehicle_states[tid] == "Moving" else (
                0, 0, 255)
            if is_locked:
                color = (0, 200, 0) if vehicle_states[tid] == "Moving" else (
                    0, 0, 200)

            p1 = (int(x - w_b/2), int(y - h_b/2))
            p2 = (int(x + w_b/2), int(y + h_b/2))
            cv2.rectangle(frame, p1, p2, color, 2)

            label = f"ID:{tid} {vehicle_states[tid]}"
            # Debug info: T(Translation) vs D(Deformation)
            debug_info = f"T/D:{(translation/deformation if deformation > 0 else 0):.1f}, Motion: {motion_scores[tid]}"

            cv2.putText(frame, label, (p1[0], p1[1]-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{debug_info}", (p1[0], p1[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if is_locked and (state_locked_until[tid] - frame_idx) > (LOCK_DURATION_FRAMES - 30):
                cv2.putText(
                    frame, "QUERY...", (p1[0], p1[1]-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Shape Filter Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
