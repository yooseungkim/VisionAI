import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
video_name = "parkinglot2.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)

# 1. Calibration Settings
REAL_WIDTH = 2.5
REAL_HEIGHT = 5.0
DEST_POINTS = np.array([
    [0, 0], [REAL_WIDTH, 0], [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
], dtype=np.float32)

# 2. Filtering Parameters
CONF_THRESHOLD = 0.3
SMOOTH_WINDOW = 5
SPEED_THRESH = 2.0
SHAPE_STABILITY_RATIO = 0.4

# [NEW] Occlusion Parameters
OCCLUSION_IOA_THRESH = 0.3  # 30% 이상 가려지면 페널티
OCCLUSION_PENALTY = 2.0     # 가려짐 감지 시 점수를 2.0씩 강제 차감 (강력한 억제)

# State Logic
MAX_SCORE = 300
START_THRESH = 35
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 45
MIN_TOTAL_MOVE = 0.8
STOP_DECAY_RATE = 0.4

output_path = f"results/processed_occlusion_filter_{video_name}"

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

# --- Helper Functions & Class ---


def get_box_area(box):
    return box[2] * box[3]  # w * h


def calculate_intersection_ratio(boxA, boxB):
    """
    boxA가 boxB에 의해 얼마나 가려졌는지(비율) 계산
    Ratio = Intersection / Area_of_A (IoU가 아님, IoA임)
    box format: x_center, y_center, w, h
    """
    # Convert center-wh to xyxy
    xA_1, yA_1 = boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2
    xA_2, yA_2 = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2

    xB_1, yB_1 = boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2
    xB_2, yB_2 = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2

    # Intersection coordinates
    x_inter1 = max(xA_1, xB_1)
    y_inter1 = max(yA_1, yB_1)
    x_inter2 = min(xA_2, xB_2)
    y_inter2 = min(yA_2, yB_2)

    inter_w = max(0, x_inter2 - x_inter1)
    inter_h = max(0, y_inter2 - y_inter1)

    inter_area = inter_w * inter_h
    area_A = boxA[2] * boxA[3]

    if area_A == 0:
        return 0
    return inter_area / area_A


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
model = YOLO("yolo11n-seg.pt")

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
prev_positions = {}
prev_box_dims = {}
motion_scores = defaultdict(int)
vehicle_states = defaultdict(lambda: "Stopped")
state_locked_until = defaultdict(int)
start_positions = {}

w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*'avc1'), int(FPS), (w, h))

frame_idx = 0
print(
    f"Processing... Occlusion Thresh: {OCCLUSION_IOA_THRESH}, Penalty: {OCCLUSION_PENALTY}")

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
        valid_indices = [i for i, conf in enumerate(
            raw_confs) if conf >= CONF_THRESHOLD]

        # Prepare Data for this frame
        # 상호 비교를 위해 이번 프레임의 모든 객체 정보를 리스트에 담습니다.
        current_frame_objects = []

        for idx in valid_indices:
            tid = raw_ids[idx]
            raw_box = raw_boxes[idx]

            # Smoothing
            smooth_box = smoothers[tid].update_and_get_avg(raw_box)
            x, y, w_b, h_b = smooth_box

            # BEV Transform
            pts = np.array([[[x, y + h_b / 2]]], dtype=np.float32)
            trans_pt = cv2.perspectiveTransform(pts, matrix).flatten()
            rx, ry = trans_pt[0], trans_pt[1]

            # Speed Calc
            speed_kmh = 0.0
            if tid in prev_positions:
                prev_rx, prev_ry = prev_positions[tid]
                dist_step = math.sqrt((rx - prev_rx)**2 + (ry - prev_ry)**2)
                speed_kmh = dist_step * FPS * 3.6
            prev_positions[tid] = (rx, ry)

            # Shape Logic
            is_bodily_motion = False
            deformation = 0.0
            translation = 0.0
            if tid in prev_box_dims:
                prev_w, prev_h, prev_cx, prev_cy = prev_box_dims[tid]
                deformation = abs(w_b - prev_w) + abs(h_b - prev_h)
                translation = math.sqrt((x - prev_cx)**2 + (y - prev_cy)**2)
                if speed_kmh > 10.0:
                    is_bodily_motion = True
                elif translation > (deformation * SHAPE_STABILITY_RATIO):
                    is_bodily_motion = True
            else:
                is_bodily_motion = True
            prev_box_dims[tid] = (w_b, h_b, x, y)

            # Store info for cross-comparison
            obj_data = {
                'tid': tid,
                'box': smooth_box,  # x, y, w, h
                'rx': rx, 'ry': ry,
                'speed': speed_kmh,
                'is_bodily': is_bodily_motion,
                'area': w_b * h_b
            }
            current_frame_objects.append(obj_data)

        # --- Process All Objects with Interaction Logic ---
        for obj in current_frame_objects:
            tid = obj['tid']
            box = obj['box']
            speed_kmh = obj['speed']
            is_bodily_motion = obj['is_bodily']

            # 1. Basic Motion Check
            is_moving_basic = (speed_kmh > SPEED_THRESH) and is_bodily_motion

            # 2. [NEW] Occlusion Logic (가려짐 검사)
            is_occluded = False

            for other_obj in current_frame_objects:
                if tid == other_obj['tid']:
                    continue  # 자기 자신 제외

                # 조건 1: 상대방이 나보다 크다 (Box Area 비교)
                if other_obj['area'] > obj['area']:
                    # 조건 2: 상대방이 움직이고 있다 (속도 체크)
                    if other_obj['speed'] > SPEED_THRESH:
                        # 조건 3: 상대방이 나를 30% 이상 덮었다
                        overlap_ratio = calculate_intersection_ratio(
                            box, other_obj['box'])
                        if overlap_ratio > OCCLUSION_IOA_THRESH:
                            is_occluded = True
                            # 디버깅용 시각화를 위해 상태 저장
                            obj['occluder'] = other_obj['tid']
                            break  # 하나라도 나를 덮고 지나가면 가려진 것임

            # 3. Score Update
            if is_occluded:
                # 가려졌으면 움직임 점수를 강하게 깎음 (Penalty)
                motion_scores[tid] -= OCCLUSION_PENALTY
                # 시작 위치도 리셋 (가려짐에 의한 좌표 튐 방지)
                if tid in start_positions:
                    del start_positions[tid]

            elif is_moving_basic:
                motion_scores[tid] += 1
                if tid not in start_positions:
                    start_positions[tid] = (obj['rx'], obj['ry'])
            else:
                motion_scores[tid] -= STOP_DECAY_RATE
                if motion_scores[tid] <= 0 and tid in start_positions:
                    del start_positions[tid]

            motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

            # 4. State Logic (Locked, Thresholds)
            current_state = vehicle_states[tid]
            is_locked = frame_idx < state_locked_until[tid]
            new_state = current_state

            if not is_locked:
                if current_state == "Stopped":
                    if motion_scores[tid] >= START_THRESH:
                        total_dist = 0
                        if tid in start_positions:
                            sx, sy = start_positions[tid]
                            total_dist = math.sqrt(
                                (obj['rx'] - sx)**2 + (obj['ry'] - sy)**2)

                        if total_dist >= MIN_TOTAL_MOVE:
                            new_state = "Moving"
                            print(f"[LVLM QUERY] ID {tid} Moving!")
                            state_locked_until[tid] = frame_idx + \
                                LOCK_DURATION_FRAMES
                else:
                    if motion_scores[tid] <= STOP_THRESH:
                        new_state = "Stopped"
                        state_locked_until[tid] = frame_idx + \
                            LOCK_DURATION_FRAMES

            vehicle_states[tid] = new_state

            # 5. Visualization
            color = (0, 255, 0) if vehicle_states[tid] == "Moving" else (
                0, 0, 255)
            if is_locked:
                color = (0, 200, 0) if vehicle_states[tid] == "Moving" else (
                    0, 0, 200)

            # 가려짐 감지되면 보라색으로 표시 (디버깅용)
            if is_occluded:
                color = (255, 0, 255)

            x, y, w_b, h_b = box
            p1 = (int(x - w_b/2), int(y - h_b/2))
            p2 = (int(x + w_b/2), int(y + h_b/2))
            cv2.rectangle(frame, p1, p2, color, 2)

            label = f"ID:{tid} {vehicle_states[tid]}"

            # Debug Info
            debug_txt = f"Sc:{motion_scores[tid]:.0f}"
            if is_occluded:
                debug_txt += f" OCC(by {obj.get('occluder')})"
            elif tid in start_positions:
                sx, sy = start_positions[tid]
                d = math.sqrt((obj['rx'] - sx)**2 + (obj['ry'] - sy)**2)
                debug_txt += f" D:{d:.1f}m"

            cv2.putText(frame, label, (p1[0], p1[1]-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, debug_txt, (p1[0], p1[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if is_locked and (state_locked_until[tid] - frame_idx) > (LOCK_DURATION_FRAMES - 30):
                cv2.putText(
                    frame, "QUERY...", (p1[0], p1[1]-45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Occlusion Filter Tracking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
