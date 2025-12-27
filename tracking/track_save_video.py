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
from collections import deque
from ultralytics import YOLO

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    OUTPUT_DIR = "results"
    MODEL_WEIGHTS = "yolo11m-seg.pt"
    CLASSES_TO_TRACK = [0, 2, 3, 5, 7] 
    
    BATCH_SIZE = 128          
    TARGET_WIDTH = 1280      
    TARGET_HEIGHT = 960      
    SAVE_WIDTH = 720
    SAVE_HEIGHT = 480
    
    FRAME_STRIDE = 3         
    VIDEO_SAVE_STRIDE = 1    

    # [Calibration & Zone] (CH06)
    # CALIB_SRC_PTS = np.array([(459, 835), (714, 760), (976, 819), (668, 919)], dtype=np.float32)
    # ILLEGAL_ZONES = [np.array([(1927, 584), (7, 1157), (250, 1749), (2307, 646)], dtype=np.int32)]

    # [Calibration & Zone] (CH07)
    CALIB_SRC_PTS = np.array([(1179, 836), (1440, 795), (1645, 979), (1283, 1046)], dtype=np.float32)
    ILLEGAL_ZONES = [np.array([(861, 1764), (9, 1794), (9, 1171),(779, 1083)], dtype=np.float32),
                            np.array([(828, 560), (1116, 551), (2583, 1160), (2550, 1913), (1816, 1936)], dtype=np.float32)]

     # [Calibration & Zone] (CH08)
    # CALIB_SRC_PTS = np.array([(741, 1117), (1323, 934), (2295, 1594), (1675, 1936)], dtype=np.float32)
    # ILLEGAL_ZONES = [np.array([(1935, 505),(12, 732), (88, 1243), (2244, 614)], dtype=np.int32), np.array([(741, 1117), (1323, 934), (2295, 1594), (1675, 1936)], dtype=np.float32)]


    CALIB_DST_PTS = np.array([[0, 0], [2.5, 0], [2.5, 5.0], [0, 5.0]], dtype=np.float32)

    ILLEGAL_LIMIT_SEC = 20
    EVENT_PADDING_FRAMES = 75
    
    COORD_WINDOW_SEC = 3.0      
    STD_THRESH_METER = 1.5    
    STATE_WINDOW_SEC = 3.0      
    CONFIRM_RATIO = 0.95         

# ==========================================
# 2. Helper Classes
# ==========================================
class VideoCaptureThread:
    def __init__(self, path, queue_size=256):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise ValueError(f"Cannot open video: {path}")
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()
        print(f"[*] Video Loaded: {self.orig_width}x{self.orig_height} @ {self.fps}fps")

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
        trans = cv2.perspectiveTransform(pts, self.matrix)
        return trans[0][0][0], trans[0][0][1]
    
    def is_in_zone(self, pixel_pt):
        for i, zone in enumerate(Config.ILLEGAL_ZONES):
            if cv2.pointPolygonTest(zone, pixel_pt, False) >= 0:
                return True
        return False

# [MODIFIED] Logger accepts metadata (risk info)
class BufferedEventLogger:
    def __init__(self, filepath, fps, padding=25):
        self.filepath = filepath
        self.fps = fps
        self.padding = padding
        self.active_events = {} 
        with open(self.filepath, 'w') as f: pass 

    def update(self, frame_idx, event_type, actors, metadata=None):
        actor_ids = tuple(sorted([a.id for a in actors]))
        key = (event_type, actor_ids)

        if key in self.active_events:
            self.active_events[key]['end_frame'] = max(self.active_events[key]['end_frame'], frame_idx)
            self.active_events[key]['last_seen_frame'] = max(self.active_events[key]['last_seen_frame'], frame_idx)
            self.active_events[key]['actors_data'] = self._serialize_actors(actors)
            # Update metadata if risk is higher
            if metadata:
                current_risk = self.active_events[key].get('metadata', {}).get('risk_level', 0)
                new_risk = metadata.get('risk_level', 0)
                if new_risk > current_risk:
                    self.active_events[key]['metadata'] = metadata
        else:
            self.active_events[key] = {
                'start_frame': frame_idx,
                'end_frame': frame_idx,
                'last_seen_frame': frame_idx,
                'actors_data': self._serialize_actors(actors),
                'metadata': metadata if metadata else {}
            }

    def flush(self, current_frame, force_all=False):
        TOLERANCE = Config.FRAME_STRIDE * 10 
        keys_to_remove = []

        for key, data in self.active_events.items():
            if force_all or (current_frame - data['last_seen_frame'] > TOLERANCE):
                self._write_to_file(key[0], data)
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.active_events[k]

    def _write_to_file(self, event_type, data):
        final_start = max(0, data['start_frame'] - self.padding)
        final_end = data['end_frame'] + self.padding
        time_stamp= f"{final_start // 1500}m {(final_start % 1500) // 25}s"
        duration = (final_end - final_start) / self.fps

        entry = {
            "timestamp": time_stamp,
            "event": event_type,
            "start_frame": int(final_start),
            "end_frame": int(final_end),
            "real_start": int(data['start_frame']),
            "real_end": int(data['end_frame']),
            "duration_sec": float(f"{duration:.2f}"),
            "risk_analysis": data.get('metadata', {}), # [NEW] Log Risk Info
            "actors": data['actors_data']
        }
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _serialize_actors(self, actors):
        data = []
        for actor in actors:
            bbox = [float(x) for x in actor.orig_box] if actor.orig_box is not None else []
            center = [int(actor.curr_pixel_orig[0]), int(actor.curr_pixel_orig[1])]
            cls_name = "Person" if actor.is_person else "Vehicle"
            data.append({
                "id": int(actor.id), "type": cls_name, 
                "bbox": bbox, "center": center
            })
        return data

# ==========================================
# 3. Kalman Filter & Risk Analyzer
# ==========================================
class KalmanCenterTracker:
    def __init__(self, init_pos):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5 
        self.kf.statePre = np.array([[init_pos[0]], [init_pos[1]], [0], [0]], np.float32)
        self.kf.statePost = np.array([[init_pos[0]], [init_pos[1]], [0], [0]], np.float32)

    def update(self, meas_pos):
        self.kf.predict()
        self.kf.correct(np.array([[meas_pos[0]], [meas_pos[1]]], np.float32))
        return (self.kf.statePost[0, 0], self.kf.statePost[1, 0])

# [NEW] Risk Analyzer Module
class RiskAnalyzer:
    @staticmethod
    def calculate_risk(actor_a, actor_b):
        """
        Calculates risk level (1-4) based on distance and approach status.
        Returns: (risk_level, distance_m, description)
        """
        # 1. Current Distance
        pos_a = np.array(actor_a.curr_meter)
        pos_b = np.array(actor_b.curr_meter)
        dist_curr = np.linalg.norm(pos_a - pos_b)
        
        # 2. Previous Distance (for delta)
        # If history not available, assume constant distance (safe)
        if actor_a.prev_meter is None or actor_b.prev_meter is None:
            dist_prev = dist_curr
        else:
            prev_a = np.array(actor_a.prev_meter)
            prev_b = np.array(actor_b.prev_meter)
            dist_prev = np.linalg.norm(prev_a - prev_b)

        # 3. Determine Direction
        # Delta < 0: Approaching (Closer), Delta > 0: Diverging (Away)
        delta_dist = dist_curr - dist_prev
        is_approaching = delta_dist < -0.05 # Use slight threshold for noise
        
        direction_str = "Approaching" if is_approaching else "Diverging"
        if abs(delta_dist) < 0.05: direction_str = "Static"

        # 4. Determine Risk Level (1-4)
        # Level 4: < 3m AND Approaching
        # Level 3: < 6m AND Approaching
        # Level 2: < 6m (Diverging) OR < 10m (Approaching)
        # Level 1: > 10m OR Safe
        
        risk_level = 1
        
        if dist_curr < 3.0 and is_approaching:
            risk_level = 4
        elif dist_curr < 6.0 and is_approaching:
            risk_level = 3
        elif dist_curr < 6.0: # Very close but diverging
            risk_level = 2
        elif dist_curr < 10.0 and is_approaching:
            risk_level = 2
        
        return {
            "risk_level": risk_level,
            "distance_m": float(f"{dist_curr:.2f}"),
            "direction": direction_str,
            "delta_m": float(f"{delta_dist:.3f}")
        }

# ==========================================
# 4. Tracked Actor
# ==========================================
class TrackedActor:
    def __init__(self, track_id, class_id):
        self.id = track_id
        self.class_id = int(class_id)
        self.is_person = (self.class_id == 0)
        self.is_vehicle = (self.class_id in [2, 3, 5, 7])
        
        self.curr_pixel_orig = (0, 0)
        self.curr_meter = (0, 0)
        self.prev_meter = None # [NEW] For Risk Calculation
        
        self.orig_box = None
        self.mask_contour_resized = None 
        self.committed_state = "Stopped" 
        
        self.kf = None
        self.coord_window = None 
        self.std_val = 0.0 
        
        self.state_window = None
        self.illegal_timer = 0

    def update(self, box_orig, mask_contour_resized, geo_engine, time_interval, current_frame_idx, fps):
        self.orig_box = box_orig
        self.mask_contour_resized = mask_contour_resized 
        raw_cx, raw_cy = float(box_orig[0]), float(box_orig[1])
        
        if self.kf is None: self.kf = KalmanCenterTracker((raw_cx, raw_cy))
        if self.coord_window is None:
            c_len = int((Config.COORD_WINDOW_SEC * fps) / Config.FRAME_STRIDE)
            self.coord_window = deque(maxlen=max(c_len, 2))
            s_len = int((Config.STATE_WINDOW_SEC * fps) / Config.FRAME_STRIDE)
            self.state_window = deque(maxlen=max(s_len, 2))

        # Kalman Update
        smooth_x, smooth_y = self.kf.update((raw_cx, raw_cy))
        self.curr_pixel_orig = (int(smooth_x), int(smooth_y))

        # Meter Conversion
        meter_x, meter_y = geo_engine.pixel_to_meter((smooth_x, smooth_y))
        
        # [NEW] Save Previous Meter Position before updating current
        # (Used for risk analysis in the next step, stored for next frame)
        # Note: We update 'prev_meter' at the end or track strictly? 
        # Here we just store the value from the *previous* call.
        # But since we just calculated new meter, let's keep the OLD one in prev_meter for one cycle logic outside.
        # Actually, best to update prev_meter at end of update cycle? 
        # Easier: Let RiskAnalyzer access self.curr_meter and self.prev_meter.
        # So we update prev_meter NOW with the OLD curr_meter.
        if self.curr_meter != (0,0):
            self.prev_meter = self.curr_meter
        else:
            self.prev_meter = (meter_x, meter_y) # First init

        self.curr_meter = (meter_x, meter_y)
        
        # Motion Logic
        self.coord_window.append((meter_x, meter_y))
        
        raw_state = "Stopped"
        if len(self.coord_window) >= int(self.coord_window.maxlen * 0.3):
            coords = np.array(self.coord_window)
            std_devs = np.std(coords, axis=0)
            self.std_val = np.linalg.norm(std_devs)
            
            if self.std_val > Config.STD_THRESH_METER:
                raw_state = "Moving"
        
        # State Confirmation
        self.state_window.append(raw_state)
        
        confirmed_events = []
        if len(self.state_window) >= int(self.state_window.maxlen * 0.5):
            moving_cnt = self.state_window.count("Moving")
            total = len(self.state_window)
            moving_ratio = moving_cnt / total
            
            if self.committed_state == "Stopped" and moving_ratio >= Config.CONFIRM_RATIO:
                self.committed_state = "Moving"
                if self.is_vehicle: # [MODIFIED] Only Vehicle triggers Start event
                    confirmed_events.append(("Vehicle_Started", current_frame_idx))
            
            elif self.committed_state == "Moving" and (1.0 - moving_ratio) >= Config.CONFIRM_RATIO:
                self.committed_state = "Stopped"
                if self.is_vehicle: # [MODIFIED] Only Vehicle triggers Stop event
                    confirmed_events.append(("Vehicle_Stopped", current_frame_idx))

        # Illegal Parking
        is_illegal_now = False
        if self.is_vehicle and self.committed_state == "Stopped":
            if geo_engine.is_in_zone(self.curr_pixel_orig):
                self.illegal_timer += time_interval
                if self.illegal_timer > Config.ILLEGAL_LIMIT_SEC:
                    is_illegal_now = True
            else:
                self.illegal_timer = 0
        else:
            self.illegal_timer = 0 
        
        return confirmed_events, is_illegal_now

# ==========================================
# 5. Main System Controller
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
        self.temp_path = os.path.join(Config.OUTPUT_DIR, f"videos/temp_{base_name}.mp4")
        self.final_path = os.path.join(Config.OUTPUT_DIR, f"videos/{base_name}_result.mp4")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] Initializing on {device}...")
        self.model = YOLO(Config.MODEL_WEIGHTS).to(device)
        self.geo = GeometryEngine()
        self.actors = {} 
        self.reported_illegal_ids = set()

        self.cap_thread = VideoCaptureThread(self.source_path)
        self.logger = BufferedEventLogger(self.log_path, self.cap_thread.fps, padding=Config.EVENT_PADDING_FRAMES)
        
        self.orig_w = self.cap_thread.orig_width
        self.orig_h = self.cap_thread.orig_height
        self.scale_x = self.orig_w / Config.TARGET_WIDTH
        self.scale_y = self.orig_h / Config.TARGET_HEIGHT

        self.out_fps = self.cap_thread.fps / (Config.FRAME_STRIDE * Config.VIDEO_SAVE_STRIDE)
        
        self.temp_path_abs = os.path.abspath(self.temp_path)
        self.final_path_abs = os.path.abspath(self.final_path)
        
        self.out = cv2.VideoWriter(self.temp_path_abs, cv2.VideoWriter_fourcc(*'mp4v'), self.out_fps, (Config.SAVE_WIDTH, Config.SAVE_HEIGHT))

    def run(self):
        print(f"Processing Resolution: {Config.TARGET_WIDTH}x{Config.TARGET_HEIGHT}")
        print(f"Saving Resolution:     {Config.SAVE_WIDTH}x{Config.SAVE_HEIGHT} @ {self.out_fps:.2f} FPS")
        print(f"Risk Analysis:         Delta-Distance based (Level 1-4)")
        
        frame_cnt_global = 0
        start_t = time.time()
        time_interval = Config.FRAME_STRIDE / self.cap_thread.fps
        
        batch_frames_resized = []
        batch_frame_indices = []

        try:
            while True:
                frame = self.cap_thread.read()
                if frame is None:
                    if self.cap_thread.stopped: break
                    continue 

                current_idx = frame_cnt_global
                frame_cnt_global += 1

                if current_idx % Config.FRAME_STRIDE != 0:
                    continue

                frame_resized = cv2.resize(frame, (Config.TARGET_WIDTH, Config.TARGET_HEIGHT))
                batch_frames_resized.append(frame_resized)
                batch_frame_indices.append(current_idx)

                if len(batch_frames_resized) == Config.BATCH_SIZE:
                    self._process_batch(batch_frames_resized, batch_frame_indices, time_interval)
                    batch_frames_resized = []
                    batch_frame_indices = []
                    
                    if current_idx % (Config.BATCH_SIZE * 10) == 0:
                        fps_real = current_idx / (time.time() - start_t)
                        print(f"Processed {current_idx} frames. Speed: {fps_real:.2f} FPS")

            if batch_frames_resized:
                self._process_batch(batch_frames_resized, batch_frame_indices, time_interval)
            
            self.logger.flush(frame_cnt_global, force_all=True)

        except KeyboardInterrupt: print("Stopped.")
        finally: self.cleanup()

    def _process_batch(self, frames, indices, time_interval):
        results = self.model.track(
            frames, persist=True, classes=Config.CLASSES_TO_TRACK, verbose=False, retina_masks=False 
        )

        for i, result in enumerate(results):
            frame_idx = indices[i]
            frame_viz = frames[i].copy() 
            
            active_vehicle_actors = []
            people_actors = []
            mask_overlay = frame_viz.copy()

            if result.boxes and result.boxes.id is not None:
                boxes_resized = result.boxes.xywh.cpu().numpy()
                boxes_original = boxes_resized.copy()
                boxes_original[:, 0] *= self.scale_x 
                boxes_original[:, 1] *= self.scale_y 
                boxes_original[:, 2] *= self.scale_x 
                boxes_original[:, 3] *= self.scale_y 
                
                ids = result.boxes.id.int().cpu().tolist()
                clss = result.boxes.cls.int().cpu().tolist()
                
                masks_resized = [None] * len(ids)
                if result.masks is not None:
                    raw_masks = result.masks.xy 
                    for k, m in enumerate(raw_masks):
                        if len(m) > 0: masks_resized[k] = m
                
                for k, tid in enumerate(ids):
                    if tid not in self.actors: self.actors[tid] = TrackedActor(tid, clss[k])
                    actor = self.actors[tid]
                    
                    events_with_frame, is_illegal_now = actor.update(
                        boxes_original[k], masks_resized[k], self.geo, time_interval, frame_idx, self.cap_thread.fps
                    )
                    
                    # Log Events
                    for evt_name, start_frame in events_with_frame:
                        print(f"[EVENT] {evt_name} - ID {tid} [{start_frame // 1500}m {(start_frame% 1500) // 25}s]")
                        self.logger.update(start_frame, evt_name, [actor])

                    # Log Illegal Parking
                    if is_illegal_now:
                        already_reported = tid in self.reported_illegal_ids
                        event_key = ("Illegal_Parking", (int(tid),))
                        is_ongoing = event_key in self.logger.active_events
                        
                        if not already_reported:
                            self.logger.update(frame_idx, "Illegal_Parking", [actor])
                            self.reported_illegal_ids.add(tid)
                            print(f"[VIOLATION] NEW Vehicle {tid} Illegal Parking Detected!")
                        elif is_ongoing:
                            self.logger.update(frame_idx, "Illegal_Parking", [actor])

                    if actor.is_vehicle and actor.committed_state == "Moving": active_vehicle_actors.append(actor)
                    if actor.is_person: people_actors.append(actor) # Person added even if 'Moving' status not logged
                    
                    self.draw_actor(frame_viz, actor, mask_overlay, boxes_resized[k])

            # [Risk Analysis: Vehicle - Pedestrian]
            if active_vehicle_actors and people_actors:
                highest_risk = 0
                risk_meta = {}
                # Check all pairs
                for v in active_vehicle_actors:
                    for p in people_actors:
                        analysis = RiskAnalyzer.calculate_risk(v, p)
                        if analysis['risk_level'] > highest_risk:
                            highest_risk = analysis['risk_level']
                            risk_meta = analysis
                
                # Only log if there is minimal risk
                if highest_risk > 1:
                    self.logger.update(frame_idx, "Danger_Pedestrian_Interaction", active_vehicle_actors + people_actors, risk_meta)

            # [Risk Analysis: Vehicle - Vehicle]
            if len(active_vehicle_actors) >= 2:
                highest_risk = 0
                risk_meta = {}
                for idx1 in range(len(active_vehicle_actors)):
                    for idx2 in range(idx1 + 1, len(active_vehicle_actors)):
                        v1 = active_vehicle_actors[idx1]
                        v2 = active_vehicle_actors[idx2]
                        analysis = RiskAnalyzer.calculate_risk(v1, v2)
                        if analysis['risk_level'] > highest_risk:
                            highest_risk = analysis['risk_level']
                            risk_meta = analysis
                
                if highest_risk > 1:
                    self.logger.update(frame_idx, "Danger_Vehicle_Interaction", active_vehicle_actors, risk_meta)
            
            self.logger.flush(frame_idx)

            if i % Config.VIDEO_SAVE_STRIDE == 0:
                for zone in Config.ILLEGAL_ZONES:
                    zone_resized = (zone / [self.scale_x, self.scale_y]).astype(np.int32)
                    cv2.polylines(frame_viz, [zone_resized], True, (0,0,255), 2)
                
                cv2.addWeighted(mask_overlay, 0.4, frame_viz, 0.6, 0, frame_viz)
                
                frame_save = cv2.resize(frame_viz, (Config.SAVE_WIDTH, Config.SAVE_HEIGHT))
                self.out.write(frame_save)

    def draw_actor(self, frame, actor, mask_overlay, box_resized):
        color = (200,200,200)
        if actor.is_vehicle:
            if actor.committed_state == "Moving": color = (0,255,0)
            elif actor.illegal_timer > Config.ILLEGAL_LIMIT_SEC: color = (0,0,255)
            elif actor.committed_state == "Stopped": color = (0,165,255)
        if actor.is_person: color = (255,255,0)

        if actor.mask_contour_resized is not None:
            cv2.fillPoly(mask_overlay, [actor.mask_contour_resized.astype(np.int32)], color)
            cv2.polylines(frame, [actor.mask_contour_resized.astype(np.int32)], True, color, 2)
        
        cx, cy = int(box_resized[0]), int(box_resized[1])
        label = f"ID:{actor.id} {actor.committed_state}"
        if actor.id in self.reported_illegal_ids: label += " VIOLATION"
        cv2.putText(frame, label, (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    def cleanup(self):
        print("[*] Releasing resources...")
        if hasattr(self, 'cap_thread'): self.cap_thread.release()
        if hasattr(self, 'out'): self.out.release()
        if os.path.exists(self.temp_path_abs) and os.path.getsize(self.temp_path_abs) > 0: 
            self.convert_h264()

    def convert_h264(self):
        print("[*] Converting to H.264...")
        cmd = ["ffmpeg", "-y", "-i", self.temp_path_abs, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-an", self.final_path_abs]
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Saved to: {self.final_path_abs}")
            if os.path.exists(self.temp_path_abs): os.remove(self.temp_path_abs)
        except Exception as e: print(f"[ERROR] FFmpeg: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking Surveillance System (Final)")
    parser.add_argument('--source', type=str, required=True, help='Path to the input video file')
    args = parser.parse_args()
    try:
        sys = ParkingSurveillanceSystem(source_path=args.source)
        sys.run()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")

