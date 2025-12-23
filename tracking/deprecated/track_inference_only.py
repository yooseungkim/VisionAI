import os
import sys
import time
import threading
import math
import cv2
import json
import numpy as np
import torch
import argparse 
from queue import Queue, Empty
from ultralytics import YOLO

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    OUTPUT_DIR = "results"
    MODEL_WEIGHTS = "yolo11n-seg.pt"
    CLASSES_TO_TRACK = [0, 2, 3, 5, 7]
    
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 960
    FRAME_STRIDE = 2 # 짝수 프레임만 처리

    # Calibration & Zone
    CALIB_SRC_PTS = np.array([(459, 835), (714, 760), (976, 819), (668, 919)], dtype=np.float32)
    CALIB_DST_PTS = np.array([[0, 0], [2.5, 0], [2.5, 5.0], [0, 5.0]], dtype=np.float32)
    ILLEGAL_ZONE_POLY = np.array([[0,0]]) 

    # Logic Parameters
    MAX_SCORE = 200           
    START_THRESH = 20        
    STOP_THRESH = 5          
    
    SCORE_INC = 4            
    SCORE_DEC_NORMAL = 1     
    SCORE_DEC_OCCLUDED = 0.5 
    
    MOVE_THRESH_METER = 0.25 
    
    GLITCH_AREA_RATIO = 1.3
    GLITCH_SPEED_LIMIT = 50.0

    ILLEGAL_LIMIT_SEC = 5.0
    LOCK_DURATION = 10 
    
    EVENT_PADDING_FRAMES = 25
    
    # [NEW] 최소 유지 시간 (3초)
    MIN_STATE_DURATION_SEC = 3.0

# ==========================================
# 2. Helper Classes
# ==========================================
class VideoCaptureThread:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()
        print(f"[*] Video Loaded: {self.width}x{self.height} @ {self.fps}fps")

    def _reader(self):
        while not self.stopped:
            if self.q.full(): time.sleep(0.01); continue
            ret, frame = self.cap.read()
            if not ret: self.stopped = True; break
            self.q.put(frame)

    def read(self):
        if self.stopped and self.q.empty(): return None
        try: return self.q.get(timeout=0.5)
        except Empty: return None

    def release(self):
        self.stopped = True; self.t.join(); self.cap.release()

class GeometryEngine:
    def __init__(self):
        self.matrix = cv2.getPerspectiveTransform(Config.CALIB_SRC_PTS, Config.CALIB_DST_PTS)
    def pixel_to_meter(self, pixel_pt):
        pts = np.array([[[pixel_pt[0], pixel_pt[1]]]], dtype=np.float32)
        trans = cv2.perspectiveTransform(pts, self.matrix).flatten()
        return trans[0], trans[1]
    def is_in_zone(self, pixel_pt):
        return cv2.pointPolygonTest(Config.ILLEGAL_ZONE_POLY, pixel_pt, False) >= 0

# ==========================================
# Buffered Event Logger
# ==========================================
class BufferedEventLogger:
    def __init__(self, filepath, fps, padding=25):
        self.filepath = filepath
        self.fps = fps
        self.padding = padding
        self.active_events = {} 
        with open(self.filepath, 'w') as f: pass 

    def update(self, frame_idx, event_type, actors):
        actor_ids = tuple(sorted([a.id for a in actors]))
        key = (event_type, actor_ids)

        if key in self.active_events:
            # 기존 이벤트 연장: end_frame만 현재(혹은 지정된) 프레임으로 갱신
            # 주의: Delayed Commit으로 인해 frame_idx가 과거일 수 있으나, end_frame은 확장해야 함
            self.active_events[key]['end_frame'] = max(self.active_events[key]['end_frame'], frame_idx)
            self.active_events[key]['last_seen_frame'] = max(self.active_events[key]['last_seen_frame'], frame_idx)
            self.active_events[key]['actors_data'] = self._serialize_actors(actors)
        else:
            # 새로운 이벤트 등록
            self.active_events[key] = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'last_seen_frame': frame_idx,
                'actors_data': self._serialize_actors(actors)
            }

    def flush(self, current_frame, force_all=False):
        TOLERANCE = Config.FRAME_STRIDE * 10 
        keys_to_remove = []

        for key, data in self.active_events.items():
            # 마지막 업데이트로부터 시간이 많이 지났으면 파일에 쓰고 제거
            if force_all or (current_frame - data['last_seen_frame'] > TOLERANCE):
                self._write_to_file(key[0], data)
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.active_events[k]

    def _write_to_file(self, event_type, data):
        final_start = max(0, data['start_frame'] - self.padding)
        final_end = data['end_frame'] + self.padding
        duration = (final_end - final_start) / self.fps

        entry = {
            "event": event_type,
            "start_frame": int(final_start),
            "end_frame": int(final_end),
            "real_start": int(data['start_frame']),
            "real_end": int(data['end_frame']),
            "duration_sec": float(f"{duration:.2f}"),
            "actors": data['actors_data']
        }
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _serialize_actors(self, actors):
        data = []
        for actor in actors:
            bbox = [float(x) for x in actor.box] if actor.box is not None else []
            center = [int(actor.curr_pixel[0]), int(actor.curr_pixel[1])]
            cls_name = "Person" if actor.is_person else "Vehicle"
            data.append({
                "id": int(actor.id), "type": cls_name, 
                "bbox": bbox, "center": center
            })
        return data

def calculate_ioa(boxA, boxB):
    xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
    yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
    xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
    yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxB_area = boxB[2] * boxB[3]
    return interArea / boxB_area if boxB_area > 0 else 0

# ==========================================
# 3. Tracked Actor (Modified)
# ==========================================
class TrackedActor:
    def __init__(self, track_id, class_id):
        self.id = track_id
        self.class_id = int(class_id)
        self.is_person = (self.class_id == 0)
        self.is_vehicle = (self.class_id in [2, 3, 5, 7])
        
        self.curr_pixel = (0, 0)
        self.curr_meter = (0, 0)
        self.prev_meter = None
        self.prev_area = 0
        self.box = None
        
        # [State Logic]
        self.state = "Stopped"          # 현재 프레임의 추정 상태
        self.motion_score = 0
        self.state_locked_until = 0     # (기존 Score 로직용)
        
        # [NEW: State Confirmation Logic]
        self.committed_state = "Stopped" # 3초 이상 유지되어 확정된 상태
        self.state_change_frame = 0      # self.state가 바뀐 시점

        self.illegal_timer = 0

    def update(self, box, mask_area, geo_engine, time_interval, is_occluded, current_frame_idx, fps):
        self.box = box
        cx = int(box[0])
        cy = int(box[1])
        self.curr_pixel = (cx, cy)
        self.curr_meter = geo_engine.pixel_to_meter((cx, cy))
        
        # 1. Raw Move Detection
        is_glitch = False
        speed_val = 0.0
        if self.prev_meter is not None:
            speed_val = math.sqrt((self.curr_meter[0]-self.prev_meter[0])**2 + 
                                  (self.curr_meter[1]-self.prev_meter[1])**2)
            if speed_val > Config.GLITCH_SPEED_LIMIT: is_glitch = True
            if self.prev_area > 0 and mask_area > 0:
                ratio = max(mask_area, self.prev_area) / min(mask_area, self.prev_area)
                if ratio > Config.GLITCH_AREA_RATIO: is_glitch = True

        is_moving_raw = (not is_glitch) and (speed_val > Config.MOVE_THRESH_METER)

        # 2. Score Update
        if is_glitch: self.motion_score -= Config.SCORE_DEC_NORMAL
        elif is_occluded: self.motion_score -= Config.SCORE_DEC_OCCLUDED
        elif is_moving_raw: self.motion_score += Config.SCORE_INC
        else: self.motion_score -= Config.SCORE_DEC_NORMAL
        self.motion_score = max(0, min(self.motion_score, Config.MAX_SCORE))

        # 3. Instant State Transition (Score-based)
        is_locked = current_frame_idx < self.state_locked_until
        if not is_locked:
            if self.state == "Stopped" or self.state == "Init":
                if self.motion_score >= Config.START_THRESH:
                    self.state = "Moving"
                    self.state_locked_until = current_frame_idx + Config.LOCK_DURATION
                    # [NEW] 상태 변경 시점 기록
                    if self.state != self.committed_state and self.state_change_frame == 0:
                         self.state_change_frame = current_frame_idx
            elif self.state == "Moving":
                if self.motion_score <= Config.STOP_THRESH:
                    self.state = "Stopped"
                    self.state_locked_until = current_frame_idx + Config.LOCK_DURATION
                    # [NEW] 상태 변경 시점 기록
                    if self.state != self.committed_state and self.state_change_frame == 0:
                         self.state_change_frame = current_frame_idx

        # 4. [NEW] Delayed State Confirmation (3초 필터)
        confirmed_events = []
        
        # 상태가 다시 원복되었다면 (예: Moving -> Stopped 찍고 1초만에 다시 Moving)
        if self.state == self.committed_state:
            self.state_change_frame = 0 # 타이머 리셋
        else:
            # 상태가 다름. 얼마나 지났는지 체크
            if self.state_change_frame == 0:
                self.state_change_frame = current_frame_idx # 초기화 누락 방지
            
            duration_frames = current_frame_idx - self.state_change_frame
            min_frames = Config.MIN_STATE_DURATION_SEC * fps
            
            if duration_frames >= min_frames:
                # 3초 지남 -> 상태 확정!
                if self.committed_state == "Stopped" and self.state == "Moving":
                    # 3초 전에 출발했음이 이제야 확인됨
                    confirmed_events.append(("Vehicle_Started", self.state_change_frame))
                elif self.committed_state == "Moving" and self.state == "Stopped":
                    confirmed_events.append(("Vehicle_Stopped", self.state_change_frame))
                
                # 확정 상태 업데이트
                self.committed_state = self.state
                self.state_change_frame = 0

        # 5. Illegal Parking (Stopped 상태가 확정된 상태에서만 체크)
        is_illegal_now = False
        if self.is_vehicle and self.committed_state == "Stopped":
            if geo_engine.is_in_zone(self.curr_pixel):
                self.illegal_timer += time_interval
                if self.illegal_timer > Config.ILLEGAL_LIMIT_SEC:
                    is_illegal_now = True
            else:
                self.illegal_timer = 0
        else:
            self.illegal_timer = 0 # 움직이면 리셋

        self.prev_meter = self.curr_meter
        self.prev_area = mask_area
        
        # 이벤트 이름과 발생(시작) 프레임 튜플 리스트 반환
        return confirmed_events, is_illegal_now

# ==========================================
# 4. Main System Controller
# ==========================================
class ParkingSurveillanceSystem:
    def __init__(self, source_path):
        self.source_path = source_path
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Video not found: {self.source_path}")

        video_name = os.path.basename(self.source_path)
        base_name = os.path.splitext(video_name)[0]
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self.log_path = os.path.join(Config.OUTPUT_DIR, f"logs/events_{base_name}.jsonl")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] Initializing on {device}...")
        self.model = YOLO(Config.MODEL_WEIGHTS).to(device)
        self.geo = GeometryEngine()
        self.actors = {} 

        self.cap_thread = VideoCaptureThread(self.source_path)
        self.logger = BufferedEventLogger(self.log_path, self.cap_thread.fps, padding=Config.EVENT_PADDING_FRAMES)
        
        self.orig_w = self.cap_thread.width
        self.orig_h = self.cap_thread.height
        self.scale_x = self.orig_w / Config.TARGET_WIDTH
        self.scale_y = self.orig_h / Config.TARGET_HEIGHT

    def run(self):
        print(f"Processing Started: Resizing to {Config.TARGET_WIDTH}x{Config.TARGET_HEIGHT}")
        print(f"Stride: Process every {Config.FRAME_STRIDE} frames")
        print(f"Filtering: Ignore state changes < {Config.MIN_STATE_DURATION_SEC} sec")
        print(f"Events will be logged to: {self.log_path}")
        
        frame_cnt_global = 0
        start_t = time.time()
        time_interval = Config.FRAME_STRIDE / self.cap_thread.fps
        fps = self.cap_thread.fps

        try:
            while True:
                frame = self.cap_thread.read()
                if frame is None:
                    if self.cap_thread.stopped: break
                    continue 

                if frame_cnt_global % Config.FRAME_STRIDE != 0:
                    frame_cnt_global += 1
                    continue

                frame_resized = cv2.resize(frame, (Config.TARGET_WIDTH, Config.TARGET_HEIGHT))
                results = self.model.track(
                    frame_resized, 
                    persist=True, 
                    classes=Config.CLASSES_TO_TRACK, 
                    verbose=False, 
                    retina_masks=False 
                )
                
                result = results[0]
                active_vehicle_actors = []
                people_actors = []

                if result.boxes and result.boxes.id is not None:
                    boxes_resized = result.boxes.xywh.cpu().numpy()
                    boxes_original = boxes_resized.copy()
                    boxes_original[:, 0] *= self.scale_x 
                    boxes_original[:, 1] *= self.scale_y 
                    boxes_original[:, 2] *= self.scale_x 
                    boxes_original[:, 3] *= self.scale_y 
                    
                    ids = result.boxes.id.int().cpu().tolist()
                    clss = result.boxes.cls.int().cpu().tolist()
                    
                    masks_data = []
                    if result.masks is not None:
                        raw_masks = result.masks.data.cpu().numpy()
                        for m in raw_masks:
                            area = np.sum(m) * (self.scale_x * self.scale_y)
                            masks_data.append(area)
                    else:
                        masks_data = [b[2]*b[3] for b in boxes_original]

                    occlusion_map = {tid: False for tid in ids}
                    # (Occlusion check omitted for brevity)

                    for i, tid in enumerate(ids):
                        if tid not in self.actors: self.actors[tid] = TrackedActor(tid, clss[i])
                        actor = self.actors[tid]
                        
                        # [Modified Update Call]
                        events_with_frame, is_illegal_now = actor.update(
                            boxes_original[i], 
                            masks_data[i], 
                            self.geo, 
                            time_interval, 
                            occlusion_map[tid], 
                            frame_cnt_global,
                            fps # FPS 전달 (3초 계산용)
                        )
                        
                        # [Logging]
                        # event: ("Vehicle_Started", start_frame)
                        for evt_name, start_frame in events_with_frame:
                            print(f"[EVENT] {evt_name} - ID {tid} (Detected at F{frame_cnt_global}, Actual Start F{start_frame})")
                            self.logger.update(start_frame, evt_name, [actor])

                        if is_illegal_now:
                            self.logger.update(frame_cnt_global, "Illegal_Parking", [actor])

                        if actor.is_vehicle and actor.committed_state == "Moving": # 확정된 Moving만 사용
                            active_vehicle_actors.append(actor)
                        if actor.is_person: people_actors.append(actor)

                if active_vehicle_actors and people_actors:
                    self.logger.update(frame_cnt_global, "Danger_Pedestrian_Interaction", active_vehicle_actors + people_actors)

                # 현재 프레임 기준 flush (오래된 이벤트 저장)
                self.logger.flush(frame_cnt_global)

                frame_cnt_global += 1

                if frame_cnt_global % 100 == 0:
                    fps_real = frame_cnt_global / (time.time() - start_t)
                    print(f"Processed {frame_cnt_global} frames. Avg Speed: {fps_real:.2f} FPS")
            
            self.logger.flush(frame_cnt_global, force_all=True)

        except KeyboardInterrupt: print("Stopped.")
        finally: self.cleanup()

    def cleanup(self):
        print("[*] Releasing resources...")
        if hasattr(self, 'cap_thread'): self.cap_thread.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Surveillance System v2")
    parser.add_argument('--source', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    try:
        sys = ParkingSurveillanceSystem(source_path=args.source)
        sys.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")