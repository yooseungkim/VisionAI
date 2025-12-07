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
output_path = f"results/final_mask_viz_{video_name}"

# 1. Physical Settings
REAL_WIDTH = 2.5
REAL_HEIGHT = 5.0
# DEST_POINTS는 Calibration 완료 후 계산됨 (여기서는 비율만 정의)
DEST_REF_POINTS = np.array([
    [0, 0], [REAL_WIDTH, 0], [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
], dtype=np.float32)

# 2. Logic Parameters
CONF_THRESHOLD = 0.5
SMOOTH_WINDOW = 10
OCCLUSION_IOA_THRESH = 0.3
OCCLUSION_PENALTY = 1.0

# Motion Thresholds
SPEED_THRESH = 2.0
AREA_CHANGE_THRESH = 0.02

# Glitch Filter
GLITCH_AREA_RATIO = 1.3
GLITCH_SPEED_LIMIT = 30.0

# State Logic
MAX_SCORE = 300
START_THRESH = 35
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 45
MIN_TOTAL_MOVE = 0.8
STOP_DECAY_RATE = 0.4

# Re-ID & Illegal Parking
REID_SIMILARITY_THRESH = 0.80
MAX_LOST_FRAMES = 150
ILLEGAL_TIME_LIMIT_SEC = 10.0
ILLEGAL_SCORE_IGNORE = 5.0

# [NEW] Door Ding Parameters
DOOR_OPEN_RATIO = 1.10  # 평소 너비보다 10% 이상 커지면 문 열림으로 간주
DING_PROXIMITY_THRESH = 20  # 문 열린 차와 옆 차의 거리가 20픽셀 이내면 문콕 경고

# 문콕 감지용 데이터
# {tid: deque([width1, width2, ...])} -> 정지 상태일 때의 너비 평균을 구하기 위함
stopped_widths = defaultdict(lambda: deque(maxlen=30))

# --- GUI Global Variables ---
calibration_points = []
restricted_zones = []
current_polygon = []
gui_mode = "calibration"  # Start with Calibration


def mouse_callback(event, x, y, flags, param):
    global calibration_points, current_polygon, restricted_zones, gui_mode

    if gui_mode == "calibration":
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(calibration_points) < 4:
                calibration_points.append((x, y))

    elif gui_mode == "zoning":
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(current_polygon) > 2:
                restricted_zones.append(
                    np.array(current_polygon, dtype=np.int32))
                current_polygon = []
                print(f"Zone Added. Total: {len(restricted_zones)}")

# --- Helper Classes ---


class VehicleReID:
    def __init__(self):
        self.known_hists = {}
        self.lost_tracks = {}
        self.id_map = {}
        self.next_visual_id = 1

    def get_histogram(self, img, mask_contour, box):
        x, y, w, h = map(int, box)
        roi = img[max(0, y):min(img.shape[0], y+h),
                  max(0, x):min(img.shape[1], x+w)]
        if roi.size == 0:
            return None
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        roi_cnt = mask_contour - [x, y]
        cv2.drawContours(mask, [roi_cnt.astype(np.int32)], -1, 255, -1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [18, 20], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, yolo_id, frame_idx, img, mask_contour, box):
        if yolo_id in self.id_map:
            visual_id = self.id_map[yolo_id]
            hist = self.get_histogram(img, mask_contour, box)
            if hist is not None:
                self.known_hists[visual_id] = hist
            return visual_id

        curr_hist = self.get_histogram(img, mask_contour, box)
        if curr_hist is None:
            return self.next_visual_id

        best_match_id = -1
        max_sim = -1.0
        ids_to_remove = []
        for lost_id, (lost_hist, lost_frame) in self.lost_tracks.items():
            if frame_idx - lost_frame > MAX_LOST_FRAMES:
                ids_to_remove.append(lost_id)
                continue
            sim = cv2.compareHist(curr_hist, lost_hist, cv2.HISTCMP_CORREL)
            if sim > max_sim:
                max_sim = sim
                best_match_id = lost_id
        for i in ids_to_remove:
            del self.lost_tracks[i]

        if max_sim > REID_SIMILARITY_THRESH:
            self.id_map[yolo_id] = best_match_id
            del self.lost_tracks[best_match_id]
            self.known_hists[best_match_id] = curr_hist
            return best_match_id

        visual_id = self.next_visual_id
        self.next_visual_id += 1
        self.id_map[yolo_id] = visual_id
        self.known_hists[visual_id] = curr_hist
        return visual_id

    def cleanup_lost_tracks(self, current_active_visual_ids, frame_idx):
        active_set = set(current_active_visual_ids)
        known_ids = list(self.known_hists.keys())
        for vid in known_ids:
            if vid not in active_set:
                self.lost_tracks[vid] = (self.known_hists[vid], frame_idx)
                del self.known_hists[vid]

    def get_permanently_lost_ids(self, frame_idx):
        ids = []
        for lost_id, (_, lost_frame) in self.lost_tracks.items():
            if frame_idx - lost_frame > MAX_LOST_FRAMES:
                ids.append(lost_id)
        return ids


class BoxSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update_and_get_avg(self, box):
        self.window.append(box)
        return np.mean(self.window, axis=0)


def calculate_intersection_ratio(boxA, boxB):
    xA_1, yA_1 = boxA[0]-boxA[2]/2, boxA[1]-boxA[3]/2
    xA_2, yA_2 = boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2
    xB_1, yB_1 = boxB[0]-boxB[2]/2, boxB[1]-boxB[3]/2
    xB_2, yB_2 = boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2
    x_inter1 = max(xA_1, xB_1)
    y_inter1 = max(yA_1, yB_1)
    x_inter2 = min(xA_2, xB_2)
    y_inter2 = min(yA_2, yB_2)
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    area_A = boxA[2] * boxA[3]
    return inter_area / area_A if area_A > 0 else 0


def check_point_in_zones(point, zones):
    for zone in zones:
        if cv2.pointPolygonTest(zone, point, False) >= 0:
            return True
    return False


# --- Main Logic ---
cap = cv2.VideoCapture(video_path)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
model = YOLO("yolo11n-seg.pt")

# ==========================================
# Phase 1: Calibration GUI (Restored)
# ==========================================
cv2.namedWindow("Setup")
cv2.setMouseCallback("Setup", mouse_callback)
success, first_frame = cap.read()
if not success:
    exit()

while True:
    img = first_frame.copy()

    # HUD
    cv2.rectangle(img, (10, 10), (450, 180), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (450, 180), (255, 255, 255), 2)
    cv2.putText(img, "STEP 1: CALIBRATION", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, "Click 4 corners of a parking slot", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    status_color = (0, 255, 0) if len(calibration_points) == 4 else (0, 0, 255)
    cv2.putText(img, f"Selected: {len(calibration_points)}/4",
                (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if len(calibration_points) == 4:
        cv2.putText(img, "Press [SPACE] to Next Step", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Draw Points
    for pt in calibration_points:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    if len(calibration_points) > 1:
        cv2.polylines(img, [np.array(calibration_points)],
                      False, (0, 255, 255), 2)

    cv2.imshow("Setup", img)
    key = cv2.waitKey(10)
    if key == ord(' '):
        if len(calibration_points) == 4:
            gui_mode = "zoning"
            break
    elif key == ord('q'):
        exit()

# ==========================================
# Phase 2: Zoning GUI
# ==========================================
while True:
    img = first_frame.copy()

    # Draw Calibration (Reference)
    cv2.polylines(
        img, [np.array(calibration_points, dtype=np.int32)], True, (100, 100, 100), 1)

    # Draw Zones
    overlay = img.copy()
    for zone in restricted_zones:
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    for zone in restricted_zones:
        cv2.polylines(img, [zone], True, (0, 0, 255), 2)

    if len(current_polygon) > 0:
        cv2.polylines(
            img, [np.array(current_polygon, dtype=np.int32)], False, (0, 255, 0), 2)
        for pt in current_polygon:
            cv2.circle(img, pt, 3, (0, 255, 0), -1)

    # HUD
    cv2.rectangle(img, (10, 10), (400, 210), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (400, 210), (255, 255, 255), 2)
    cv2.putText(img, "STEP 2: SET ZONES", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f"Zones Created: {len(restricted_zones)}",
                (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "[L-Click]: Add Point", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "[R-Click]: Close Zone", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(img, "Press [SPACE] to Start", (30, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Setup", img)
    key = cv2.waitKey(10)
    if key == ord(' '):
        cv2.destroyWindow("Setup")
        break
    elif key == ord('q'):
        exit()

# Matrix Calculation
source_points = np.array(calibration_points, dtype=np.float32)
matrix = cv2.getPerspectiveTransform(source_points, DEST_REF_POINTS)

# Trackers
reid_system = VehicleReID()
smoothers = defaultdict(lambda: BoxSmoother(window_size=SMOOTH_WINDOW))
prev_positions = {}
prev_mask_area = {}
motion_scores = defaultdict(int)
vehicle_states = defaultdict(lambda: "Stopped")
state_locked_until = defaultdict(int)
start_positions = {}
illegal_timers = defaultdict(int)

w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*'avc1'), int(FPS), (w, h))
frame_idx = 0
alert_limit = int(ILLEGAL_TIME_LIMIT_SEC * FPS)

print("Processing Started...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    results = model.track(frame, persist=True, classes=[
                          2, 3, 5, 7], verbose=False, retina_masks=True)
    result = results[0]

    # Create separate overlay for masks (for blending)
    mask_overlay = frame.copy()

    # Draw Zones on Frame (Bottom Layer)
    zone_overlay = frame.copy()
    for zone in restricted_zones:
        cv2.fillPoly(zone_overlay, [zone], (0, 0, 150))
    cv2.addWeighted(zone_overlay, 0.2, frame, 0.8, 0, frame)
    for zone in restricted_zones:
        cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

    current_visual_ids = []
    current_frame_objects = []

    door_open_events = []  # (Attacker ID, Victim ID)

    # 1단계: 정지한 차량들의 평소 너비 학습 & 문 열림 감지
    for obj in current_frame_objects:
        tid = obj['tid']
        box = obj['box']  # x, y, w, h
        mask = obj['mask']
        state = vehicle_states[tid]

        # 마스크 기반의 정확한 너비 계산 (Bounding Box Width보다 정확함)
        # 마스크의 가장 왼쪽 점과 오른쪽 점의 차이
        if mask is not None:
            x_coords = mask[:, 0]
            mask_width = x_coords.max() - x_coords.min()
        else:
            mask_width = box[2]  # 마스크 없으면 bbox w 사용

        obj['mask_width'] = mask_width  # 나중에 비교 위해 저장
        obj['is_door_open'] = False

        if state == "Stopped":
            # 너비 히스토리 업데이트
            width_history = stopped_widths[tid]

            if len(width_history) >= 10:
                avg_w = np.mean(width_history)

                # [조건 1] 갑자기 너비가 증가했는가? (문 열림)
                if mask_width > avg_w * DOOR_OPEN_RATIO:
                    obj['is_door_open'] = True
                    # 주의: 문이 열린 상태를 '평균'에 포함시키면 안 됨 -> append 생략
                else:
                    # 문 안 열렸으면 히스토리에 추가
                    width_history.append(mask_width)
            else:
                width_history.append(mask_width)
        else:
            # 움직이는 중에는 너비 데이터 초기화 (회전하면 너비가 바뀌므로)
            stopped_widths[tid].clear()

    # 2단계: 문콕(접촉) 판별
    # "문이 열린 차(Attacker)"와 "주변에 있는 차(Victim)" 사이의 거리 계산
    for attacker in current_frame_objects:
        if not attacker['is_door_open']:
            continue

        attacker_id = attacker['tid']
        a_box = attacker['box']  # x, y, w, h

        for victim in current_frame_objects:
            if attacker_id == victim['tid']:
                continue

            victim_id = victim['tid']
            v_box = victim['box']

            # [조건 2] 두 차량이 물리적으로 매우 가까운가?
            # Bbox 간의 거리 계산 (단순화: x축 거리만 확인, 주차장은 보통 나란히 주차하므로)
            # Attacker Right <-> Victim Left  OR  Attacker Left <-> Victim Right

            dist_x = min(abs((a_box[0] + a_box[2]/2) - (v_box[0] - v_box[2]/2)),  # A_Right - V_Left
                         # A_Left - V_Right
                         abs((a_box[0] - a_box[2]/2) - (v_box[0] + v_box[2]/2)))

            if dist_x < DING_PROXIMITY_THRESH:
                # 문콕 감지!
                door_open_events.append((attacker_id, victim_id))
                print(
                    f"[ALERT] Door Ding Detected! ID {attacker_id} -> ID {victim_id}")

    # -----------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------
    for obj in current_frame_objects:
        tid = obj['tid']
        box = obj['box']

        # 기본 시각화 (기존 코드 유지)
        # ...

        # 문 열림 표시
        if obj.get('is_door_open'):
            cv2.putText(frame, "DOOR OPEN", (int(box[0]), int(box[1]-40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # 문이 열려서 늘어난 영역 시각화 (옵션)

    # 문콕 경고선 그리기
    for att_id, vic_id in door_open_events:
        # 두 차량 사이에 빨간 번개 표시 혹은 선 연결
        att_obj = next(o for o in current_frame_objects if o['tid'] == att_id)
        vic_obj = next(o for o in current_frame_objects if o['tid'] == vic_id)

        p1 = (int(att_obj['box'][0]), int(att_obj['box'][1]))
        p2 = (int(vic_obj['box'][0]), int(vic_obj['box'][1]))

        cv2.line(frame, p1, p2, (0, 0, 255), 3)
        cv2.putText(frame, "DING!", (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # ... (Write frame & Show) ...

    if result.boxes and result.boxes.id is not None:
        raw_boxes = result.boxes.xywh.cpu().numpy()
        raw_ids = result.boxes.id.int().cpu().tolist()
        raw_confs = result.boxes.conf.cpu().tolist()
        masks_xy = result.masks.xy if result.masks is not None else [
            None]*len(raw_boxes)
        valid_indices = [i for i, conf in enumerate(
            raw_confs) if conf >= CONF_THRESHOLD]

        for idx in valid_indices:
            box = raw_boxes[idx]
            yolo_id = raw_ids[idx]
            mask_cnt = masks_xy[idx]

            if mask_cnt is None or len(mask_cnt) == 0:
                bx, by, bw, bh = box
                mask_cnt = np.array(
                    [[bx-bw/2, by-bh/2], [bx+bw/2, by-bh/2], [bx+bw/2, by+bh/2], [bx-bw/2, by+bh/2]])

            mask_area_pixels = cv2.contourArea(mask_cnt.astype(np.float32))
            visual_id = reid_system.update(
                yolo_id, frame_idx, frame, mask_cnt, box)
            current_visual_ids.append(visual_id)
            smooth_box = smoothers[visual_id].update_and_get_avg(box)
            x, y, w_b, h_b = smooth_box

            pts = np.array([[[x, y + h_b / 2]]], dtype=np.float32)
            trans_pt = cv2.perspectiveTransform(pts, matrix).flatten()
            rx, ry = trans_pt[0], trans_pt[1]

            speed_kmh = 0.0
            if visual_id in prev_positions:
                dist_step = math.sqrt(
                    (rx - prev_positions[visual_id][0])**2 + (ry - prev_positions[visual_id][1])**2)
                speed_kmh = dist_step * FPS * 3.6
            prev_positions[visual_id] = (rx, ry)

            is_glitch = False
            area_change_ratio = 0.0
            if visual_id in prev_mask_area:
                prev_area = prev_mask_area[visual_id]
                if prev_area > 0:
                    area_change_ratio = abs(
                        mask_area_pixels - prev_area) / prev_area
                    if area_change_ratio > (GLITCH_AREA_RATIO - 1.0):
                        is_glitch = True
            if speed_kmh > GLITCH_SPEED_LIMIT:
                is_glitch = True

            is_moving_speed = False
            is_moving_area = False
            if not is_glitch:
                if speed_kmh > SPEED_THRESH:
                    is_moving_speed = True
                if area_change_ratio > AREA_CHANGE_THRESH:
                    is_moving_area = True

            prev_mask_area[visual_id] = mask_area_pixels

            current_frame_objects.append({
                'tid': visual_id, 'box': smooth_box, 'rx': rx, 'ry': ry,
                'speed': speed_kmh, 'area_ratio': area_change_ratio,
                'is_moving_speed': is_moving_speed, 'is_moving_area': is_moving_area,
                'area': w_b*h_b, 'is_glitch': is_glitch, 'mask': mask_cnt
            })

        # Logic & Visualization Loop
        for obj in current_frame_objects:
            tid = obj['tid']
            box = obj['box']
            mask = obj['mask']

            # Logic
            is_moving_detected = obj['is_moving_speed'] or obj['is_moving_area']
            is_occluded = False
            for other in current_frame_objects:
                if tid == other['tid']:
                    continue
                if other['area'] > obj['area'] and (other['is_moving_speed'] or other['is_moving_area']):
                    if calculate_intersection_ratio(box, other['box']) > OCCLUSION_IOA_THRESH:
                        is_occluded = True
                        break

            if obj['is_glitch']:
                motion_scores[tid] -= STOP_DECAY_RATE
            elif is_occluded:
                motion_scores[tid] -= OCCLUSION_PENALTY
                if tid in start_positions:
                    del start_positions[tid]
            elif is_moving_detected:
                motion_scores[tid] += 1
                if tid not in start_positions:
                    start_positions[tid] = (obj['rx'], obj['ry'])
            else:
                motion_scores[tid] -= STOP_DECAY_RATE
                if motion_scores[tid] <= 0 and tid in start_positions:
                    del start_positions[tid]
            motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

            is_locked = frame_idx < state_locked_until[tid]
            if not is_locked:
                if vehicle_states[tid] == "Stopped":
                    if motion_scores[tid] >= START_THRESH:
                        total_dist = 0
                        if tid in start_positions:
                            sx, sy = start_positions[tid]
                            total_dist = math.sqrt(
                                (obj['rx'] - sx)**2 + (obj['ry'] - sy)**2)
                        if total_dist >= MIN_TOTAL_MOVE:
                            vehicle_states[tid] = "Moving"
                            state_locked_until[tid] = frame_idx + \
                                LOCK_DURATION_FRAMES
                            illegal_timers[tid] = 0
                else:
                    if motion_scores[tid] <= STOP_THRESH:
                        vehicle_states[tid] = "Stopped"
                        state_locked_until[tid] = frame_idx + \
                            LOCK_DURATION_FRAMES

            # Illegal Logic
            warning_level = 0
            x, y, w_b, h_b = box
            foot_point = (int(x), int(y + h_b / 2))
            in_zone = check_point_in_zones(foot_point, restricted_zones)

            if in_zone:
                is_trying_to_move = motion_scores[tid] > ILLEGAL_SCORE_IGNORE
                if vehicle_states[tid] == "Stopped" and not is_trying_to_move:
                    illegal_timers[tid] += 1
                    warning_level = 1
                    if illegal_timers[tid] > alert_limit:
                        warning_level = 2
                        if illegal_timers[tid] == alert_limit + 1:
                            print(
                                f"[ALERT] ID {tid} Illegal Parking Confirmed!")
                else:
                    illegal_timers[tid] = 0
            else:
                illegal_timers[tid] = 0

            # --- Visualization (Mask) ---
            # Determine Color
            if warning_level == 2:  # Illegal
                # Blink Red/Yellow
                color = (0, 0, 255) if (frame_idx % 10 < 5) else (0, 255, 255)
                status_txt = f"ILLEGAL! {illegal_timers[tid]/FPS:.1f}s"
            elif warning_level == 1:  # Warning
                color = (0, 165, 255)  # Orange
                status_txt = f"Wait {illegal_timers[tid]/FPS:.1f}s"
            elif vehicle_states[tid] == "Moving":
                color = (0, 255, 0)  # Green
                status_txt = "Moving"
            else:
                color = (200, 200, 200)  # Stopped (Gray)
                status_txt = "Stopped"

            if is_occluded:
                color = (255, 0, 255)
                status_txt = "Occ"

            # Draw Mask on Overlay
            cv2.fillPoly(mask_overlay, [mask.astype(np.int32)], color)

            # Draw Contour on Frame (Solid Line)
            cv2.polylines(frame, [mask.astype(np.int32)], True, color, 2)

            # Labels
            p1 = (int(x - w_b/2), int(y - h_b/2))
            cv2.putText(frame, f"ID:{tid} {status_txt}",
                        (p1[0], p1[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Sc:{motion_scores[tid]:.0f}", (
                p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Blend Mask Overlay
    cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)

    reid_system.cleanup_lost_tracks(current_visual_ids, frame_idx)
    for lid in reid_system.get_permanently_lost_ids(frame_idx):
        if lid in illegal_timers:
            del illegal_timers[lid]

    out.write(frame)
    cv2.imshow("Mask Viz System", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
