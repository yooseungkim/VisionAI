import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# Calibration & Zone (기존 동일)
    CALIB_SRC_PTS = np.array([(459, 835), (714, 760), (976, 819), (668, 919)], dtype=np.float32)
    CALIB_DST_PTS = np.array([[0, 0], [2.5, 0], [2.5, 5.0], [0, 5.0]], dtype=np.float32)
    ILLEGAL_ZONE_POLY = np.array([(1887, 582), (6, 1152), (263, 1722), (2278, 641)], dtype=np.int32)
# --- Configuration ---
video_name = "parking1.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)

# 1. Calibration
REAL_WIDTH = 2.5
REAL_HEIGHT = 5.0
DEST_POINTS = np.array([
    [0, 0], [REAL_WIDTH, 0], [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
], dtype=np.float32)

# 2. Basic Parameters
CONF_THRESHOLD = 0.4
SMOOTH_WINDOW = 5
SPEED_THRESH = 2.0
SHAPE_STABILITY_RATIO = 0.4
OCCLUSION_IOA_THRESH = 0.3
OCCLUSION_PENALTY = 2.0

# [NEW] Glitch Filter Parameters (급격한 변화 방지)
GLITCH_SIZE_RATIO = 1.3    # 전 프레임 대비 면적이 30% 이상 변하면 노이즈로 간주
GLITCH_SPEED_LIMIT = 30.0  # 순간 속도가 30km/h를 넘으면 박스가 튄 것으로 간주

# State Logic
MAX_SCORE = 200
START_THRESH = 25
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 60
MIN_TOTAL_MOVE = 0.8
STOP_DECAY_RATE = 0.4

# Re-ID & Illegal Parking
REID_SIMILARITY_THRESH = 0.80
MAX_LOST_FRAMES = 150
ILLEGAL_TIME_LIMIT_SEC = 10.0
ILLEGAL_SCORE_IGNORE = 5.0

output_path = f"results/processed_final_glitch_filter_{video_name}"

# --- Global Variables for GUI ---
calibration_points = []
restricted_zones = []
current_polygon = []
gui_mode = "calibration"


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

# GUI Setup
cv2.namedWindow("Setup")
cv2.setMouseCallback("Setup", mouse_callback)
success, first_frame = cap.read()
if not success:
    exit()

print("Phase 1: Click 4 Calibration Points -> Space")
while True:
    img = first_frame.copy()
    cv2.putText(img, "STEP 1: CALIBRATION", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    for pt in calibration_points:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    if len(calibration_points) > 1:
        cv2.polylines(img, [np.array(calibration_points)],
                      False, (0, 255, 255), 2)

    if len(calibration_points) == 4:
        cv2.putText(img, "Press SPACE to Next Step", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Setup", img)
    if cv2.waitKey(10) == ord(' '):
        if len(calibration_points) == 4:
            gui_mode = "zoning"
            break
    elif cv2.waitKey(1) == ord('q'):
        exit()

print("Phase 2: Draw Zones -> Space")
while True:
    img = first_frame.copy()
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

    cv2.putText(img, f"Zones Created: {len(restricted_zones)}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, "L-Click: Add / R-Click: Close", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "SPACE: Start / 'c': Clear Last", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Setup", img)
    if cv2.waitKey(10) == ord(' '):
        cv2.destroyWindow("Setup")
        break
    elif cv2.waitKey(1) == ord('c'):
        if restricted_zones:
            restricted_zones.pop()
    elif cv2.waitKey(1) == ord('q'):
        exit()

source_points = np.array(calibration_points, dtype=np.float32)
matrix = cv2.getPerspectiveTransform(source_points, DEST_POINTS)

# Trackers
reid_system = VehicleReID()
smoothers = defaultdict(lambda: BoxSmoother(window_size=SMOOTH_WINDOW))
prev_positions = {}
prev_box_dims = {}
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

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1

    results = model.track(frame, persist=True, classes=[
                          2, 3, 5, 7], verbose=False)
    result = results[0]

    # Draw Zones
    overlay = frame.copy()
    for zone in restricted_zones:
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    for zone in restricted_zones:
        cv2.polylines(frame, [zone], True, (0, 0, 255), 2)
    cv2.polylines(frame, [np.array(calibration_points,
                  dtype=np.int32)], True, (100, 100, 100), 1)

    current_visual_ids = []
    current_frame_objects = []

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

            # Re-ID
            mask_cnt = masks_xy[idx]
            if mask_cnt is None or len(mask_cnt) == 0:
                bx, by, bw, bh = box
                mask_cnt = np.array(
                    [[bx-bw/2, by-bh/2], [bx+bw/2, by-bh/2], [bx+bw/2, by+bh/2], [bx-bw/2, by+bh/2]])
            visual_id = reid_system.update(
                yolo_id, frame_idx, frame, mask_cnt, box)
            current_visual_ids.append(visual_id)

            # Smoothing
            smooth_box = smoothers[visual_id].update_and_get_avg(box)
            x, y, w_b, h_b = smooth_box

            # BEV Transform
            pts = np.array([[[x, y + h_b / 2]]], dtype=np.float32)
            trans_pt = cv2.perspectiveTransform(pts, matrix).flatten()
            rx, ry = trans_pt[0], trans_pt[1]

            # Speed Calc
            speed_kmh = 0.0
            if visual_id in prev_positions:
                dist_step = math.sqrt(
                    (rx - prev_positions[visual_id][0])**2 + (ry - prev_positions[visual_id][1])**2)
                speed_kmh = dist_step * FPS * 3.6
            prev_positions[visual_id] = (rx, ry)

            # --- [NEW] Glitch Detection (Too Fast or Too Deformed) ---
            is_glitch = False

            # 1. Size Glitch Check
            if visual_id in prev_box_dims:
                pw, ph, _, _ = prev_box_dims[visual_id]
                prev_area = pw * ph
                curr_area = w_b * h_b
                if prev_area > 0:
                    area_ratio = max(prev_area, curr_area) / \
                        min(prev_area, curr_area)
                    if area_ratio > GLITCH_SIZE_RATIO:
                        is_glitch = True  # 30% 이상 급격한 크기 변화

            # 2. Speed Glitch Check (Impossible Speed in Parking Lot)
            if speed_kmh > GLITCH_SPEED_LIMIT:
                is_glitch = True

            # Shape Filter Logic (if not glitch)
            is_bodily = False
            if not is_glitch:
                if visual_id in prev_box_dims:
                    pw, ph, px, py = prev_box_dims[visual_id]
                    deform = abs(w_b - pw) + abs(h_b - ph)
                    trans = math.sqrt((x - px)**2 + (y - py)**2)
                    if speed_kmh > 10.0:
                        is_bodily = True
                    elif trans > (deform * SHAPE_STABILITY_RATIO):
                        is_bodily = True
                else:
                    is_bodily = True

            # 만약 Glitch라면 이번 프레임의 정보는 저장하되, 움직임 판단에서는 배제
            prev_box_dims[visual_id] = (w_b, h_b, x, y)

            current_frame_objects.append({
                'tid': visual_id, 'box': smooth_box, 'rx': rx, 'ry': ry,
                'speed': speed_kmh, 'is_bodily': is_bodily, 'area': w_b*h_b,
                'is_glitch': is_glitch  # Pass glitch flag
            })

        # Logic Loop
        for obj in current_frame_objects:
            tid = obj['tid']
            box = obj['box']
            speed_kmh = obj['speed']
            is_bodily = obj['is_bodily']
            is_glitch = obj['is_glitch']

            # Occlusion Check
            is_occluded = False
            for other in current_frame_objects:
                if tid == other['tid']:
                    continue
                if other['area'] > obj['area'] and other['speed'] > SPEED_THRESH:
                    if calculate_intersection_ratio(box, other['box']) > OCCLUSION_IOA_THRESH:
                        is_occluded = True
                        break

            # Score Update Logic
            if is_glitch:
                # Glitch detected: Force decay (Ignore this frame's motion)
                motion_scores[tid] -= STOP_DECAY_RATE
            elif is_occluded:
                motion_scores[tid] -= OCCLUSION_PENALTY
                if tid in start_positions:
                    del start_positions[tid]
            elif (speed_kmh > SPEED_THRESH) and is_bodily:
                motion_scores[tid] += 1
                if tid not in start_positions:
                    start_positions[tid] = (obj['rx'], obj['ry'])
            else:
                motion_scores[tid] -= STOP_DECAY_RATE
                if motion_scores[tid] <= 0 and tid in start_positions:
                    del start_positions[tid]

            motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

            # State Update
            curr_state = vehicle_states[tid]
            is_locked = frame_idx < state_locked_until[tid]
            new_state = curr_state

            if not is_locked:
                if curr_state == "Stopped":
                    if motion_scores[tid] >= START_THRESH:
                        total_dist = 0
                        if tid in start_positions:
                            sx, sy = start_positions[tid]
                            total_dist = math.sqrt(
                                (obj['rx'] - sx)**2 + (obj['ry'] - sy)**2)
                        if total_dist >= MIN_TOTAL_MOVE:
                            new_state = "Moving"
                            state_locked_until[tid] = frame_idx + \
                                LOCK_DURATION_FRAMES
                            illegal_timers[tid] = 0
                else:
                    if motion_scores[tid] <= STOP_THRESH:
                        new_state = "Stopped"
                        state_locked_until[tid] = frame_idx + \
                            LOCK_DURATION_FRAMES

            vehicle_states[tid] = new_state

            # Illegal Parking
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

            # Visualization
            if warning_level == 2:
                color = (0, 0, 255) if (frame_idx % 10 < 5) else (0, 255, 255)
                status_txt = f"ILLEGAL! {illegal_timers[tid]/FPS:.1f}s"
            elif warning_level == 1:
                color = (0, 165, 255)
                status_txt = f"Wait {illegal_timers[tid]/FPS:.1f}s"
            elif vehicle_states[tid] == "Moving":
                color = (0, 255, 0)
                status_txt = "Moving"
            else:
                color = (200, 200, 200)
                status_txt = "Stopped"

            if is_occluded:
                color = (255, 0, 255)
                status_txt = "Occ"
            if is_glitch:
                status_txt += " (Glitch)"

            p1, p2 = (int(x - w_b/2), int(y - h_b/2)
                      ), (int(x + w_b/2), int(y + h_b/2))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(frame, f"ID:{tid} {status_txt}",
                        (p1[0], p1[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Sc:{motion_scores[tid]:.0f}", (
                p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    reid_system.cleanup_lost_tracks(current_visual_ids, frame_idx)
    for lid in reid_system.get_permanently_lost_ids(frame_idx):
        if lid in illegal_timers:
            del illegal_timers[lid]

    out.write(frame)
    cv2.imshow("Final Illegal Parking", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
