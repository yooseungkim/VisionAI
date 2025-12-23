import math
import os
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Configuration ---
video_name = "parking3.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)
output_path = f"results/server_mask_viz_{video_name}"

# 1. Physical Settings (REMOVED: Calibration)
# 서버 환경에서는 픽셀 단위로만 계산합니다.

# 2. Logic Parameters (Adjusted for Pixel Units)
CONF_THRESHOLD = 0.5
SMOOTH_WINDOW = 10
OCCLUSION_IOA_THRESH = 0.3
OCCLUSION_PENALTY = 1.0

# Motion Thresholds (Units: Pixels)
# 픽셀 단위로 변경됨 (해상도에 따라 조절 필요, FHD 기준 대략적 설정)
SPEED_THRESH = 2.0         # 프레임당 2픽셀 이상 움직여야 움직임으로 간주
AREA_CHANGE_THRESH = 0.02

# Glitch Filter
GLITCH_AREA_RATIO = 1.3
GLITCH_SPEED_LIMIT = 150.0 # 프레임당 150픽셀 이상 점프하면 글리치로 간주

# State Logic
MAX_SCORE = 300
START_THRESH = 35
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 45
MIN_TOTAL_MOVE = 50.0      # 상태 변경을 위해 최소 50픽셀 이상 이동해야 함
STOP_DECAY_RATE = 0.4

# Re-ID & Illegal Parking
REID_SIMILARITY_THRESH = 0.80
MAX_LOST_FRAMES = 150
ILLEGAL_TIME_LIMIT_SEC = 10.0
ILLEGAL_SCORE_IGNORE = 5.0

# --- Global Variables ---
# GUI 관련 변수 제거됨
# restricted_zones = [] # 나중에 사용할 변수

# --- Helper Classes ---

class VehicleReID:
    def __init__(self):
        self.known_hists = {}
        self.lost_tracks = {}
        self.id_map = {}
        self.next_visual_id = 1

    def get_histogram(self, img, mask_contour, box):
        x, y, w, h = map(int, box)
        # ROI 예외 처리
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
            
        roi = img[y:y+h, x:x+w]
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

# [Modification] PointSmoother for Center of Mass
class PointSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update_and_get_avg(self, point):
        self.window.append(point)
        avg = np.mean(self.window, axis=0)
        return (avg[0], avg[1])

def calculate_intersection_ratio(boxA, boxB):
    # Box IoA는 그대로 유지 (Occlusion 판단용)
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

# --- Main Logic ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video {video_path}")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
w, h = int(cap.get(3)), int(cap.get(4))

# [Modification] Load Model
# 서버 메모리 효율을 위해 yolov11n-seg 사용 권장 (또는 필요시 상위 모델)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolo11x-seg.pt").to(device)
print(f"Currently using: {model.device}")
# [Modification] Trackers & State Variables
reid_system = VehicleReID()
centroid_smoothers = defaultdict(lambda: PointSmoother(window_size=SMOOTH_WINDOW))

prev_positions = {}     # Stores (cx, cy) in pixels
prev_mask_area = {}
motion_scores = defaultdict(int)
vehicle_states = defaultdict(lambda: "Stopped")
state_locked_until = defaultdict(int)
start_positions = {}
illegal_timers = defaultdict(int)

# [Modification] GUI Setup Phase Removed
# Zone 데이터를 하드코딩하거나 파일에서 읽어와야 함
# 예시: restricted_zones = [np.array([[100,100], [200,100], ...])]
restricted_zones = [] 

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(FPS), (w, h))
frame_idx = 0
alert_limit = int(ILLEGAL_TIME_LIMIT_SEC * FPS)

print(f"Server Processing Started. Output: {output_path}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_idx += 1
    
    # Progress Logging (For Server)
    if frame_idx % 100 == 0:
        print(f"Processing Frame: {frame_idx}")

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False, retina_masks=True)
    result = results[0]

    mask_overlay = frame.copy()
    
    # [Future Use] Zone Visualization (Annotate only)
    # for zone in restricted_zones:
    #     cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

    current_visual_ids = []
    current_frame_objects = []

    if result.boxes and result.boxes.id is not None:
        raw_boxes = result.boxes.xywh.cpu().numpy()
        raw_ids = result.boxes.id.int().cpu().tolist()
        raw_confs = result.boxes.conf.cpu().tolist()
        masks_xy = result.masks.xy if result.masks is not None else [None]*len(raw_boxes)
        
        valid_indices = [i for i, conf in enumerate(raw_confs) if conf >= CONF_THRESHOLD]

        for idx in valid_indices:
            box = raw_boxes[idx] # cx, cy, w, h (Box Center)
            yolo_id = raw_ids[idx]
            mask_cnt = masks_xy[idx]

            # Mask Fallback
            if mask_cnt is None or len(mask_cnt) == 0:
                bx, by, bw, bh = box
                mask_cnt = np.array([
                    [bx-bw/2, by-bh/2], [bx+bw/2, by-bh/2], 
                    [bx+bw/2, by+bh/2], [bx-bw/2, by+bh/2]
                ])

            # [Modification] Center of Mass Calculation using Moments
            M = cv2.moments(mask_cnt.astype(np.float32))
            if M['m00'] != 0:
                raw_cx = float(M['m10'] / M['m00'])
                raw_cy = float(M['m01'] / M['m00'])
            else:
                # Fallback to Box Center if mask is weird
                raw_cx, raw_cy = box[0], box[1]

            mask_area_pixels = cv2.contourArea(mask_cnt.astype(np.float32))
            
            # Re-ID Update
            visual_id = reid_system.update(yolo_id, frame_idx, frame, mask_cnt, box)
            current_visual_ids.append(visual_id)

            # [Modification] Smooth Centroid (not Box)
            smooth_cx, smooth_cy = centroid_smoothers[visual_id].update_and_get_avg((raw_cx, raw_cy))

            # [Modification] Pixel-based Speed Calculation
            speed_pixel = 0.0
            if visual_id in prev_positions:
                prev_x, prev_y = prev_positions[visual_id]
                # Euclidean distance in pixels
                dist_pixel = math.sqrt((smooth_cx - prev_x)**2 + (smooth_cy - prev_y)**2)
                speed_pixel = dist_pixel # Pixels per frame
            
            prev_positions[visual_id] = (smooth_cx, smooth_cy)

            # Glitch Check
            is_glitch = False
            area_change_ratio = 0.0
            if visual_id in prev_mask_area:
                prev_area = prev_mask_area[visual_id]
                if prev_area > 0:
                    area_change_ratio = abs(mask_area_pixels - prev_area) / prev_area
                    if area_change_ratio > (GLITCH_AREA_RATIO - 1.0):
                        is_glitch = True
            
            # [Modification] Speed check in Pixels
            if speed_pixel > GLITCH_SPEED_LIMIT:
                is_glitch = True

            is_moving_speed = False
            is_moving_area = False
            if not is_glitch:
                if speed_pixel > SPEED_THRESH:
                    is_moving_speed = True
                if area_change_ratio > AREA_CHANGE_THRESH:
                    is_moving_area = True

            prev_mask_area[visual_id] = mask_area_pixels

            current_frame_objects.append({
                'tid': visual_id, 
                'box': box,               # Keep box for Occlusion check
                'cx': smooth_cx,          # Centroid X
                'cy': smooth_cy,          # Centroid Y
                'speed': speed_pixel, 
                'area_ratio': area_change_ratio,
                'is_moving_speed': is_moving_speed, 
                'is_moving_area': is_moving_area,
                'area': mask_area_pixels, # Use Mask Area
                'is_glitch': is_glitch, 
                'mask': mask_cnt
            })

        # Logic Loop
        for obj in current_frame_objects:
            tid = obj['tid']
            mask = obj['mask']
            
            # Logic
            is_moving_detected = obj['is_moving_speed'] or obj['is_moving_area']
            
            # Check Occlusion (Still using Box IoA for efficiency)
            is_occluded = False
            for other in current_frame_objects:
                if tid == other['tid']: continue
                if other['area'] > obj['area'] and (other['is_moving_speed'] or other['is_moving_area']):
                    if calculate_intersection_ratio(obj['box'], other['box']) > OCCLUSION_IOA_THRESH:
                        is_occluded = True
                        break

            # Scoring System
            if obj['is_glitch']:
                motion_scores[tid] -= STOP_DECAY_RATE
            elif is_occluded:
                motion_scores[tid] -= OCCLUSION_PENALTY
                if tid in start_positions: del start_positions[tid]
            elif is_moving_detected:
                motion_scores[tid] += 1
                if tid not in start_positions:
                    start_positions[tid] = (obj['cx'], obj['cy'])
            else:
                motion_scores[tid] -= STOP_DECAY_RATE
                if motion_scores[tid] <= 0 and tid in start_positions:
                    del start_positions[tid]
            
            motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

            # State Transition (Pixel based distance)
            is_locked = frame_idx < state_locked_until[tid]
            if not is_locked:
                if vehicle_states[tid] == "Stopped":
                    if motion_scores[tid] >= START_THRESH:
                        total_dist = 0
                        if tid in start_positions:
                            sx, sy = start_positions[tid]
                            # Distance in Pixels
                            total_dist = math.sqrt((obj['cx'] - sx)**2 + (obj['cy'] - sy)**2)
                        
                        if total_dist >= MIN_TOTAL_MOVE:
                            vehicle_states[tid] = "Moving"
                            state_locked_until[tid] = frame_idx + LOCK_DURATION_FRAMES
                            illegal_timers[tid] = 0
                else: # Moving
                    if motion_scores[tid] <= STOP_THRESH:
                        vehicle_states[tid] = "Stopped"
                        state_locked_until[tid] = frame_idx + LOCK_DURATION_FRAMES

            # [Modification] Illegal Logic - Commented Out but Preserved
            warning_level = 0
            # To Enable later: Uncomment below
            '''
            foot_point = (int(obj['cx']), int(obj['cy'])) # Using Centroid as foot point? Or bottom?
            # 보통 주차 판단은 바닥면(Bbox bottom center)이나 Mask의 최하단점을 씁니다.
            # 하지만 요청대로 Center of Mass를 기준으로 한다면 foot_point = (cx, cy)
            
            in_zone = False
            for zone in restricted_zones:
                 if cv2.pointPolygonTest(zone, foot_point, False) >= 0:
                     in_zone = True
                     break
            
            if in_zone:
                is_trying_to_move = motion_scores[tid] > ILLEGAL_SCORE_IGNORE
                if vehicle_states[tid] == "Stopped" and not is_trying_to_move:
                    illegal_timers[tid] += 1
                    warning_level = 1
                    if illegal_timers[tid] > alert_limit:
                        warning_level = 2
                        if illegal_timers[tid] == alert_limit + 1:
                            print(f"[ALERT] ID {tid} Illegal Parking Confirmed!")
                else:
                    illegal_timers[tid] = 0
            else:
                illegal_timers[tid] = 0
            '''

            # Visualization
            if vehicle_states[tid] == "Moving":
                color = (0, 255, 0) # Green
                status_txt = "Moving"
            else:
                color = (200, 200, 200) # Gray
                status_txt = "Stopped"

            if is_occluded:
                color = (255, 0, 255)
                status_txt = "Occ"
            
            if frame_idx % 5 == 0:
                # Draw Mask
                cv2.fillPoly(mask_overlay, [mask.astype(np.int32)], color)
                cv2.polylines(frame, [mask.astype(np.int32)], True, color, 2)
                
                # Draw Centroid
                cx, cy = int(obj['cx']), int(obj['cy'])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Labels
                cv2.putText(frame, f"ID:{tid} {status_txt}", (cx-10, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Blend Overlay
    reid_system.cleanup_lost_tracks(current_visual_ids, frame_idx)
    # [Clean up timers for lost objects]
    # for lid in reid_system.get_permanently_lost_ids(frame_idx):
    #     if lid in illegal_timers: del illegal_timers[lid]
    if frame_idx % 5 == 0:
        cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)

        out.write(frame)
    # cv2.imshow removed for server

cap.release()
out.release()
print("Processing Completed.")