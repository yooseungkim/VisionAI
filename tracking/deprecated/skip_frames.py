import math
import os
import time
import threading
from queue import Queue

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import subprocess

# --- Configuration ---
video_name = "parking3.mp4"
video_path = f"datasets/{video_name}"
os.makedirs("results", exist_ok=True)
final_output_path = f"results/temp_extreme_speed_{video_name}"
temp_output_path = f"results/extreme_speed_{video_name}"

# [성능 핵심]
BATCH_SIZE = 16          # 16장씩 GPU 처리
SKIP_LOGIC_FRAMES = 16   # 16프레임 중 1개만 로직 수행 (BATCH_SIZE와 맞추는 것 권장)
MODEL_NAME = "yolo11m-seg.pt"

# Logic Params
CONF_THRESHOLD = 0.5
SMOOTH_WINDOW = 5        # 로직 수행 빈도가 낮으므로 윈도우 줄임
REID_SIMILARITY_THRESH = 0.80
MAX_LOST_FRAMES = 300    # 스킵이 많으므로 넉넉하게

# --- Simplified Helper Classes ---
class VideoCaptureThread:
    def __init__(self, path, queue_size=128):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise ValueError(f"Open Error: {path}")
        self.q = Queue(maxsize=queue_size)
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            if self.q.full():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.q.put(frame)

    def read_batch(self, batch_size):
        frames = []
        if self.stopped and self.q.empty(): return []
        for _ in range(batch_size):
            if self.q.empty():
                if self.stopped: break
                time.sleep(0.001)
                if self.q.empty(): break
            frames.append(self.q.get())
        return frames

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

# ReID도 최소화
class SimpleReID:
    def __init__(self):
        self.known_hists = {}
        self.id_map = {}
        self.next_vid = 1
    
    def get_hist(self, img, mask, box):
        x, y, w, h = map(int, box)
        x=max(0,x); y=max(0,y); w=min(w,img.shape[1]-x); h=min(h,img.shape[0]-y)
        if w<=0 or h<=0: return None
        roi = img[y:y+h, x:x+w]
        m = np.zeros(roi.shape[:2], dtype=np.uint8)
        cnt = mask - [x,y]
        cv2.drawContours(m, [cnt.astype(np.int32)], -1, 255, -1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], m, [18, 20], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def update(self, yolo_id, img, mask, box):
        if yolo_id in self.id_map: return self.id_map[yolo_id]
        
        # 새 ID인 경우에만 무거운 연산 수행
        hist = self.get_hist(img, mask, box)
        if hist is None: return self.next_vid
        
        # 매칭 로직 (생략하거나 간소화)
        # 속도를 위해 가장 단순하게: 새 ID 부여
        # 정교한 매칭이 필요하면 여기 주석 해제하여 사용
        vid = self.next_vid
        self.next_vid += 1
        self.id_map[yolo_id] = vid
        return vid

# --- Main Setup ---
cap_thread = VideoCaptureThread(video_path)
FPS = cap_thread.fps
w, h = cap_thread.width, cap_thread.height

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_NAME).to(device)

reid = SimpleReID()
state_map = {} # {vid: "Stopped" or "Moving"}

# 저장 FPS 설정 (실시간 재생 속도 맞춤)
SAVE_FPS = FPS / BATCH_SIZE
out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), SAVE_FPS, (w, h))

print(f"🔥 EXTREME SPEED MODE STARTED | Batch: {BATCH_SIZE}")

start_time = time.time()
processed_cnt = 0

try:
    while True:
        # 1. 배치 읽기 (CPU Thread)
        batch = cap_thread.read_batch(BATCH_SIZE)
        if not batch: break
        curr_batch_len = len(batch)

        # 2. 배치 추론 (GPU) - 여기서 추적(ID 유지)은 내부적으로 다 됨
        results = model.track(batch, persist=True, verbose=False, retina_masks=True)

        # 3. [핵심] 마지막 프레임만 처리!
        # 중간 프레임은 버립니다. (파이썬 루프 삭제)
        last_idx = curr_batch_len - 1
        result = results[last_idx]
        frame = batch[last_idx]

        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            masks = result.masks.xy if result.masks is not None else []
            
            mask_overlay = frame.copy()
            
            for i, yolo_id in enumerate(ids):
                box = boxes[i]
                mask = masks[i]
                if mask is None or len(mask) == 0: continue

                # ReID (마지막 프레임에 대해서만 수행)
                vid = reid.update(yolo_id, frame, mask, box)
                
                # 시각화 (바로 그리기)
                color = (0, 255, 0) # Green
                cv2.fillPoly(mask_overlay, [mask.astype(np.int32)], color)
                cv2.polylines(frame, [mask.astype(np.int32)], True, color, 2)
                
                # 텍스트
                bx, by, bw, bh = box
                cv2.putText(frame, f"ID:{vid}", (int(bx), int(by)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)

        # 4. 저장 (1장만)
        out.write(frame)
        
        processed_cnt += curr_batch_len
        
        if processed_cnt % (BATCH_SIZE * 5) == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed_cnt} frames. FPS: {processed_cnt/elapsed:.2f}")

except KeyboardInterrupt:
    print("Stopped.")

cap_thread.release()
out.release()

print(f"✅ Temporary video saved: {temp_output_path}")

# --- [추가된 부분] FFmpeg 변환 로직 ---
print("⏳ Converting to H.264 using FFmpeg...")

if os.path.exists(temp_output_path):
    command = [
        "ffmpeg", "-y",                 # -y: 덮어쓰기 허용
        "-i", temp_output_path,         # 입력 파일
        "-vcodec", "libx264",           # H.264 코덱 사용
        "-crf", "23",                   # 화질 설정 (낮을수록 고화질, 23은 기본값)
        "-preset", "fast",              # 인코딩 속도 설정
        "-an",                          # 오디오 제거 (CCTV라 불필요)
        final_output_path               # 출력 파일
    ]
    
    try:
        # ffmpeg 실행 (로그 숨김: capture_output=True)
        subprocess.run(command, check=True)
        print(f"🎉 Conversion Complete! Saved to: {final_output_path}")
        
        # (선택사항) 임시 파일 삭제
        os.remove(temp_output_path)
        print("🗑️  Temporary file removed.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")
    except FileNotFoundError:
        print("❌ FFmpeg not found. Please install ffmpeg (sudo apt install ffmpeg).")
else:
    print("❌ Error: Temporary file not found.")