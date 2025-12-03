import math
import os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm # 진행률 표시용 (서버에서 유용)

# --- Configuration ---
video_name = "parking1.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)
output_path = f"results/processed_server_tracking_{video_name}"

# ==============================================================================
# [MATH CONFIG] Internal Calculation Points
# 속도/거리 계산을 위한 내부 좌표입니다. (GUI 없이 계산용으로만 사용)
# 실제 영상의 주차칸 좌표에 맞게 수정해주세요.
# ==============================================================================
SOURCE_POINTS = np.array([
    [450, 300],   # Top-Left
    [800, 300],   # Top-Right
    [1100, 600],  # Bottom-Right
    [200, 600]    # Bottom-Left
], dtype=np.float32)

# Real World Scale (2.5m x 5.0m)
REAL_WIDTH = 2.5 
REAL_HEIGHT = 5.0
DEST_POINTS = np.array([
    [0, 0], [REAL_WIDTH, 0], [REAL_WIDTH, REAL_HEIGHT], [0, REAL_HEIGHT]
], dtype=np.float32)

# --- Logic Parameters ---
CONF_THRESHOLD = 0.3
SMOOTH_WINDOW = 5
SPEED_THRESH = 2.0
SHAPE_STABILITY_RATIO = 0.4
OCCLUSION_IOA_THRESH = 0.3
OCCLUSION_PENALTY = 2.0

# Glitch Filter
GLITCH_SIZE_RATIO = 1.3
GLITCH_SPEED_LIMIT = 30.0

# State Logic
MAX_SCORE = 300
START_THRESH = 35
STOP_THRESH = 5
LOCK_DURATION_FRAMES = 45
MIN_TOTAL_MOVE = 0.8
STOP_DECAY_RATE = 0.4

# Re-ID Parameters
REID_SIMILARITY_THRESH = 0.80
MAX_LOST_FRAMES = 150

# --- Helper Classes ---

class VehicleReID:
    def __init__(self):
        self.known_hists = {}
        self.lost_tracks = {}
        self.id_map = {}
        self.next_visual_id = 1

    def get_histogram(self, img, mask_contour, box):
        x, y, w, h = map(int, box)
        roi = img[max(0, y):min(img.shape[0], y+h), max(0, x):min(img.shape[1], x+w)]
        if roi.size == 0: return None
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
            if hist is not None: self.known_hists[visual_id] = hist
            return visual_id

        curr_hist = self.get_histogram(img, mask_contour, box)
        if curr_hist is None: return self.next_visual_id

        best_match_id = -1; max_sim = -1.0; ids_to_remove = []
        for lost_id, (lost_hist, lost_frame) in self.lost_tracks.items():
            if frame_idx - lost_frame > MAX_LOST_FRAMES:
                ids_to_remove.append(lost_id); continue
            sim = cv2.compareHist(curr_hist, lost_hist, cv2.HISTCMP_CORREL)
            if sim > max_sim: max_sim = sim; best_match_id = lost_id
        for i in ids_to_remove: del self.lost_tracks[i]

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
    x_inter1 = max(xA_1, xB_1); y_inter1 = max(yA_1, yB_1)
    x_inter2 = min(xA_2, xB_2); y_inter2 = min(yA_2, yB_2)
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    area_A = boxA[2] * boxA[3]
    return inter_area / area_A if area_A > 0 else 0

# --- Main Logic ---
def run_processing():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolo11n-seg.pt")

    # Matrix Calculation (Internal Use)
    matrix = cv2.getPerspectiveTransform(SOURCE_POINTS, DEST_POINTS)

    # Trackers Initialization
    reid_system = VehicleReID()
    smoothers = defaultdict(lambda: BoxSmoother(window_size=SMOOTH_WINDOW))
    prev_positions = {}; prev_box_dims = {}
    motion_scores = defaultdict(int)
    vehicle_states = defaultdict(lambda: "Stopped")
    state_locked_until = defaultdict(int)
    start_positions = {}

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    
    print(f"Processing Video: {video_path}")
    print(f"Output Path: {output_path}")
    
    # tqdm for progress bar
    pbar = tqdm(total=total_frames, unit="frame")
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_idx += 1

        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        result = results[0]

        # Draw Overlay for Masks (Visualization)
        mask_overlay = frame.copy()

        current_visual_ids = []
        current_frame_objects = []

        if result.boxes and result.boxes.id is not None:
            raw_boxes = result.boxes.xywh.cpu().numpy()
            raw_ids = result.boxes.id.int().cpu().tolist()
            raw_confs = result.boxes.conf.cpu().tolist()
            masks_xy = result.masks.xy if result.masks is not None else [None]*len(raw_boxes)
            valid_indices = [i for i, conf in enumerate(raw_confs) if conf >= CONF_THRESHOLD]

            for idx in valid_indices:
                box = raw_boxes[idx]
                yolo_id = raw_ids[idx]
                mask_cnt = masks_xy[idx]
                
                # Re-ID & Smoothing
                if mask_cnt is None or len(mask_cnt) == 0:
                    bx, by, bw, bh = box
                    mask_cnt = np.array([[bx-bw/2, by-bh/2], [bx+bw/2, by-bh/2], [bx+bw/2, by+bh/2], [bx-bw/2, by+bh/2]])
                
                visual_id = reid_system.update(yolo_id, frame_idx, frame, mask_cnt, box)
                current_visual_ids.append(visual_id)

                smooth_box = smoothers[visual_id].update_and_get_avg(box)
                x, y, w_b, h_b = smooth_box
                
                # BEV Transform (Speed)
                pts = np.array([[[x, y + h_b / 2]]], dtype=np.float32)
                trans_pt = cv2.perspectiveTransform(pts, matrix).flatten()
                rx, ry = trans_pt[0], trans_pt[1]

                speed_kmh = 0.0
                if visual_id in prev_positions:
                    dist_step = math.sqrt((rx - prev_positions[visual_id][0])**2 + (ry - prev_positions[visual_id][1])**2)
                    speed_kmh = dist_step * FPS * 3.6
                prev_positions[visual_id] = (rx, ry)

                # Glitch & Shape Logic
                is_glitch = False
                if visual_id in prev_box_dims:
                    pw, ph, _, _ = prev_box_dims[visual_id]
                    area_ratio = max(pw*(ph or 1), w_b*(h_b or 1)) / (min(pw*(ph or 1), w_b*(h_b or 1)) + 1e-6)
                    if area_ratio > GLITCH_SIZE_RATIO: is_glitch = True
                if speed_kmh > GLITCH_SPEED_LIMIT: is_glitch = True

                is_bodily = False
                if not is_glitch:
                    if visual_id in prev_box_dims:
                        pw, ph, px, py = prev_box_dims[visual_id]
                        deform = abs(w_b - pw) + abs(h_b - ph)
                        trans = math.sqrt((x - px)**2 + (y - py)**2)
                        if speed_kmh > 10.0: is_bodily = True
                        elif trans > (deform * SHAPE_STABILITY_RATIO): is_bodily = True
                    else: is_bodily = True
                
                prev_box_dims[visual_id] = (w_b, h_b, x, y)

                current_frame_objects.append({
                    'tid': visual_id, 'box': smooth_box, 'rx': rx, 'ry': ry,
                    'speed': speed_kmh, 'is_bodily': is_bodily, 'area': w_b*h_b,
                    'is_glitch': is_glitch, 'mask': mask_cnt
                })

            # Logic Loop
            for obj in current_frame_objects:
                tid = obj['tid']
                box = obj['box']
                mask = obj['mask']
                
                # Occlusion
                is_occluded = False
                for other in current_frame_objects:
                    if tid == other['tid']: continue
                    if other['area'] > obj['area'] and other['speed'] > SPEED_THRESH:
                        if calculate_intersection_ratio(box, other['box']) > OCCLUSION_IOA_THRESH:
                            is_occluded = True; break
                
                # Score Update
                if obj['is_glitch']: motion_scores[tid] -= STOP_DECAY_RATE
                elif is_occluded:
                    motion_scores[tid] -= OCCLUSION_PENALTY
                    if tid in start_positions: del start_positions[tid]
                elif (obj['speed'] > SPEED_THRESH) and obj['is_bodily']:
                    motion_scores[tid] += 1
                    if tid not in start_positions: start_positions[tid] = (obj['rx'], obj['ry'])
                else:
                    motion_scores[tid] -= STOP_DECAY_RATE
                    if motion_scores[tid] <= 0 and tid in start_positions: del start_positions[tid]
                
                motion_scores[tid] = max(0, min(motion_scores[tid], MAX_SCORE))

                # State Update
                is_locked = frame_idx < state_locked_until[tid]
                if not is_locked:
                    if vehicle_states[tid] == "Stopped":
                        if motion_scores[tid] >= START_THRESH:
                            total_dist = 0
                            if tid in start_positions:
                                sx, sy = start_positions[tid]
                                total_dist = math.sqrt((obj['rx'] - sx)**2 + (obj['ry'] - sy)**2)
                            if total_dist >= MIN_TOTAL_MOVE:
                                vehicle_states[tid] = "Moving"
                                state_locked_until[tid] = frame_idx + LOCK_DURATION_FRAMES
                    else:
                        if motion_scores[tid] <= STOP_THRESH:
                            vehicle_states[tid] = "Stopped"
                            state_locked_until[tid] = frame_idx + LOCK_DURATION_FRAMES

                # Visualization (Only Moving/Stopped)
                if vehicle_states[tid] == "Moving":
                    color = (0, 255, 0) # Green
                    status_txt = "Moving"
                else:
                    color = (200, 200, 200) # Gray
                    status_txt = "Stopped"
                
                if is_occluded: status_txt = "Occluded"

                # Draw Mask
                cv2.fillPoly(mask_overlay, [mask.astype(np.int32)], color)
                cv2.polylines(frame, [mask.astype(np.int32)], True, color, 2)

                # Info Text
                p1 = (int(box[0] - box[2]/2), int(box[1] - box[3]/2))
                cv2.putText(frame, f"ID:{tid} {status_txt}", (p1[0], p1[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"{obj['speed']:.1f}km/h", (p1[0], p1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Blend Overlay
        cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)

        reid_system.cleanup_lost_tracks(current_visual_ids, frame_idx)
        
        # Save Frame
        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print("Processing Complete.")

if __name__ == "__main__":
    run_processing()