import os
import sys
import time
import threading
import subprocess
import math
import cv2
import json
import numpy as np
import torch
import argparse 
from queue import Queue, Empty
from ultralytics import YOLO

# ==========================================
# 1. Configuration (Tuned for Fast Reaction)
# ==========================================
class Config:
    # [변경] 파일 경로 관련 설정은 제거하고 로직 파라미터만 유지
    OUTPUT_DIR = "results"
    
    MODEL_WEIGHTS = "yolo11s-seg.pt"
    CLASSES_TO_TRACK = [0, 2, 3, 5, 7]
    BATCH_SIZE = 5

    # Calibration & Zone (기존 동일)
    # 주의: 영상이 바뀌면 Calibration 좌표와 Zone 좌표도 사실 해당 영상에 맞춰야 합니다.
    # 현재는 요청하신 대로 소스 경로만 동적으로 변경합니다.
    CALIB_SRC_PTS = np.array([(459, 835), (714, 760), (976, 819), (668, 919)], dtype=np.float32)
    CALIB_DST_PTS = np.array([[0, 0], [2.5, 0], [2.5, 5.0], [0, 5.0]], dtype=np.float32)
    ILLEGAL_ZONE_POLY = np.array([[0,0]]) # 불법주차구역 (필요 시 수정)

    # [Logic Parameters - Tuned]
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
        print("[*] Buffering video...")
        time.sleep(1.0)

    def _reader(self):
        while not self.stopped:
            if self.q.full(): time.sleep(0.01); continue
            ret, frame = self.cap.read()
            if not ret: self.stopped = True; break
            self.q.put(frame)

    def read_batch(self, batch_size):
        frames = []
        if self.stopped and self.q.empty(): return []
        for _ in range(batch_size):
            try: frames.append(self.q.get(timeout=0.5))
            except Empty: break
        return frames

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

class EventLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, 'w') as f: pass # Reset file
    
    def log(self, frame_idx, event_type, actors):
        actor_data = []
        for actor in actors:
            bbox = [float(x) for x in actor.box] if actor.box is not None else []
            center = [int(actor.curr_pixel[0]), int(actor.curr_pixel[1])]
            cls_name = "Person" if actor.is_person else "Vehicle"
            actor_data.append({
                "id": int(actor.id), "type": cls_name, "state": actor.state,
                "bbox": bbox, "center": center
            })
        entry = {
            "frame": int(frame_idx),
            "timestamp": f"{frame_idx/25.0:.2f}s",
            "event": event_type,
            "actors": actor_data
        }
        with open(self.filepath, 'a') as f: f.write(json.dumps(entry) + "\n")

def calculate_ioa(boxA, boxB):
    xA = max(boxA[0] - boxA[2]/2, boxB[0] - boxB[2]/2)
    yA = max(boxA[1] - boxA[3]/2, boxB[1] - boxB[3]/2)
    xB = min(boxA[0] + boxA[2]/2, boxB[0] + boxB[2]/2)
    yB = min(boxA[1] + boxA[3]/2, boxB[1] + boxB[3]/2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxB_area = boxB[2] * boxB[3]
    return interArea / boxB_area if boxB_area > 0 else 0

# ==========================================
# 3. Tracked Actor (Refined Logic)
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
        self.mask_contour = None
        
        self.state = "Stopped"
        self.motion_score = 0
        self.state_locked_until = 0
        
        self.illegal_timer = 0
        self.illegal_reported = False

    def update(self, box, mask, geo_engine, time_interval, is_occluded, current_frame_idx):
        self.box = box
        
        # 1. Position
        cx, cy = 0, 0
        mask_area = 0
        if mask is not None:
            self.mask_contour = mask
            mask_area = cv2.contourArea(mask.astype(np.float32))
            M = cv2.moments(mask.astype(np.float32))
            if M['m00'] != 0: cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            else: cx, cy = int(box[0]), int(box[1])
        else:
            cx, cy = int(box[0]), int(box[1])
            mask_area = box[2] * box[3]
            
        self.curr_pixel = (cx, cy)
        self.curr_meter = geo_engine.pixel_to_meter((cx, cy))
        
        # 2. Raw Move Detection
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

        # 3. Score Update
        if is_glitch: self.motion_score -= Config.SCORE_DEC_NORMAL
        elif is_occluded: self.motion_score -= Config.SCORE_DEC_OCCLUDED
        elif is_moving_raw: self.motion_score += Config.SCORE_INC
        else: self.motion_score -= Config.SCORE_DEC_NORMAL
        
        self.motion_score = max(0, min(self.motion_score, Config.MAX_SCORE))

        # 4. State Transition
        events = []
        prev_state = self.state
        is_locked = current_frame_idx < self.state_locked_until

        if not is_locked:
            if self.state == "Stopped" or self.state == "Init":
                if self.motion_score >= Config.START_THRESH:
                    self.state = "Moving"
                    self.state_locked_until = current_frame_idx + Config.LOCK_DURATION
                    self.illegal_timer = 0
                    self.illegal_reported = False
            elif self.state == "Moving":
                if self.motion_score <= Config.STOP_THRESH:
                    self.state = "Stopped"
                    self.state_locked_until = current_frame_idx + Config.LOCK_DURATION

        # 5. Event Generation
        if prev_state == "Stopped" and self.state == "Moving": 
            events.append("Vehicle_Started")
        if prev_state == "Moving" and self.state == "Stopped": 
            events.append("Vehicle_Stopped")

        # 6. Illegal Parking
        is_new_illegal = False
        if self.is_vehicle and self.state == "Stopped":
            if geo_engine.is_in_zone(self.curr_pixel):
                self.illegal_timer += time_interval
                if self.illegal_timer > Config.ILLEGAL_LIMIT_SEC and not self.illegal_reported:
                    is_new_illegal = True
                    self.illegal_reported = True
            else:
                self.illegal_timer = 0
                self.illegal_reported = False

        self.prev_meter = self.curr_meter
        self.prev_area = mask_area
        return events, is_new_illegal

# ==========================================
# 4. Main System Controller
# ==========================================
class ParkingSurveillanceSystem:
    def __init__(self, source_path):
        # [변경] 초기화 시 source_path를 받음
        self.source_path = source_path
        
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Video not found: {self.source_path}")

        # [변경] 입력 파일명을 기반으로 출력 경로 동적 생성
        video_name = os.path.basename(self.source_path)
        base_name = os.path.splitext(video_name)[0]
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        self.log_path = os.path.join(Config.OUTPUT_DIR, f"logs/events_{base_name}.jsonl")
        self.temp_path = os.path.join(Config.OUTPUT_DIR, f"videos/temp_{base_name}.mp4")
        self.final_path = os.path.join(Config.OUTPUT_DIR, f"videos/{base_name}_result.mp4")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] Initializing on {device}...")
        self.model = YOLO(Config.MODEL_WEIGHTS).to(device)
        self.geo = GeometryEngine()
        self.logger = EventLogger(self.log_path) # 동적 경로 사용
        self.actors = {} 

        # [변경] 동적 소스 경로 사용
        self.cap_thread = VideoCaptureThread(self.source_path)
        self.out_fps = self.cap_thread.fps / Config.BATCH_SIZE
        w, h = int(self.cap_thread.width), int(self.cap_thread.height)
        
        # [변경] 절대 경로 변환 및 Writer 초기화
        self.temp_path_abs = os.path.abspath(self.temp_path)
        self.final_path_abs = os.path.abspath(self.final_path)
        self.out = cv2.VideoWriter(self.temp_path_abs, cv2.VideoWriter_fourcc(*'mp4v'), self.out_fps, (w, h))

    def run(self):
        print(f"Processing Started: Batch Size {Config.BATCH_SIZE}")
        print(f"Events will be logged to: {self.log_path}")
        
        frame_cnt_global = 0
        start_t = time.time()
        time_interval = Config.BATCH_SIZE / self.cap_thread.fps

        try:
            while True:
                batch = self.cap_thread.read_batch(Config.BATCH_SIZE)
                if not batch: break
                
                results = self.model.track(batch, persist=True, classes=Config.CLASSES_TO_TRACK, verbose=False, retina_masks=True)

                last_idx = len(batch) - 1
                result = results[last_idx]
                frame = batch[last_idx]
                frame_cnt_global += len(batch)
                
                mask_overlay = frame.copy()
                active_vehicle_actors = []
                people_actors = []

                if result.boxes and result.boxes.id is not None:
                    boxes = result.boxes.xywh.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    clss = result.boxes.cls.int().cpu().tolist()
                    masks = result.masks.xy if result.masks is not None else [None]*len(boxes)

                    # Occlusion Check (O(N^2))
                    occlusion_map = {tid: False for tid in ids}
                    for i in range(len(ids)):
                        for j in range(len(ids)):
                            if i == j: continue
                            ioa = calculate_ioa(boxes[j], boxes[i])
                            if ioa > 0.3 and boxes[j][1] > boxes[i][1]: occlusion_map[ids[i]] = True

                    for i, tid in enumerate(ids):
                        if tid not in self.actors: self.actors[tid] = TrackedActor(tid, clss[i])
                        actor = self.actors[tid]
                        
                        events, is_new_illegal = actor.update(boxes[i], masks[i], self.geo, time_interval, occlusion_map[tid], frame_cnt_global)
                        
                        # [LOGGING]
                        if "Vehicle_Started" in events:
                            print(f"[ALERT] Vehicle {tid} STARTED (F{frame_cnt_global})")
                            self.logger.log(frame_cnt_global, "Vehicle_Started", [actor])
                        
                        if "Vehicle_Stopped" in events:
                            print(f"[ALERT] Vehicle {tid} STOPPED (F{frame_cnt_global})")
                            self.logger.log(frame_cnt_global, "Vehicle_Stopped", [actor])
                        
                        if is_new_illegal:
                            print(f"[VIOLATION] Vehicle {tid} ILLEGAL PARKING CONFIRMED")
                            self.logger.log(frame_cnt_global, "Illegal_Parking", [actor])

                        if actor.is_vehicle and actor.state == "Moving": active_vehicle_actors.append(actor)
                        if actor.is_person: people_actors.append(actor)
                        
                        self.draw_actor(frame, actor, mask_overlay)

                if active_vehicle_actors and people_actors:
                    self.logger.log(frame_cnt_global, "Danger_Pedestrian_Interaction", active_vehicle_actors + people_actors)

                cv2.polylines(frame, [Config.ILLEGAL_ZONE_POLY], True, (0,0,255), 2)
                cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)
                self.out.write(frame)

                if frame_cnt_global % (Config.BATCH_SIZE * 20) == 0:
                    fps = frame_cnt_global / (time.time() - start_t)
                    print(f"Processed {frame_cnt_global} frames. Speed: {fps:.2f} FPS")

        except KeyboardInterrupt: print("Stopped.")
        finally: self.cleanup()

    def draw_actor(self, frame, actor, mask_overlay):
        color = (200,200,200)
        if actor.is_vehicle:
            if actor.state == "Moving": color = (0,255,0)
            elif actor.illegal_timer > Config.ILLEGAL_LIMIT_SEC: color = (0,0,255)
            elif actor.state == "Stopped": color = (0,165,255)
        if actor.is_person: color = (255,255,0)

        if actor.mask_contour is not None:
            cv2.fillPoly(mask_overlay, [actor.mask_contour.astype(np.int32)], color)
            cv2.polylines(frame, [actor.mask_contour.astype(np.int32)], True, color, 2)
        
        cx, cy = actor.curr_pixel
        label = f"ID:{actor.id} {actor.state}"
        if actor.illegal_timer > Config.ILLEGAL_LIMIT_SEC: label += " ILLEGAL"
        cv2.putText(frame, label, (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    def cleanup(self):
        print("[*] Releasing resources...")
        if hasattr(self, 'cap_thread'): self.cap_thread.release()
        if hasattr(self, 'out'): self.out.release()
        # [변경] self.temp_path_abs, self.final_path_abs 사용
        if os.path.exists(self.temp_path_abs) and os.path.getsize(self.temp_path_abs) > 0: self.convert_h264()

    def convert_h264(self):
        print("[*] Converting to H.264...")
        cmd = ["ffmpeg", "-y", "-i", self.temp_path_abs, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-an", self.final_path_abs]
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Saved to: {self.final_path_abs}")
            if os.path.exists(self.temp_path_abs): os.remove(self.temp_path_abs)
        except Exception as e: print(f"[ERROR] FFmpeg: {e}")

if __name__ == "__main__":
    # [추가] Argument Parsing Logic
    parser = argparse.ArgumentParser(description="Parking Surveillance System")
    parser.add_argument('--source', type=str, required=True, help='Path to the input video file (e.g., datasets/input.mp4)')
    
    args = parser.parse_args()
    
    try:
        sys = ParkingSurveillanceSystem(source_path=args.source)
        sys.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")