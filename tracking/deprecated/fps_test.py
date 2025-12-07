import cv2
import time
import torch
from ultralytics import YOLO

# --- 설정 ---
video_path = "datasets/parking3.mp4"
model_name = "yolo11m-seg.pt" # 무거운 모델
BATCH_SIZE = 16  # 4090의 VRAM을 믿고 16장씩 한 번에 처리

# --- 준비 ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_name).to(device)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video open failed!")
    exit()

print(f"🔥 MAX PERFORMANCE TEST 🔥")
print(f"Device: {device}, Model: {model_name}, Batch: {BATCH_SIZE}")
print("Logic/Drawing/Saving disabled. Pure Inference Speed Test.")

frames_buffer = []
total_frames = 0
start_time = time.time()

try:
    while True:
        # 1. BATCH_SIZE만큼 프레임 모으기 (CPU 읽기)
        frames_buffer = []
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)
        
        if not frames_buffer:
            break
        
        current_batch_size = len(frames_buffer)
        
        # 2. 배치 추론 (GPU 한 방 처리)
        # verbose=False로 콘솔 출력 끔
        results = model(frames_buffer, verbose=False, stream=False)
        
        # (여기서 results를 파싱하는 로직이 들어가면 속도가 떨어짐)
        # 지금은 순수 추론 속도만 측정
        
        total_frames += current_batch_size
        
        # 3. 속도 모니터링
        if total_frames % (BATCH_SIZE * 5) == 0:
            elapsed = time.time() - start_time
            fps = total_frames / elapsed
            print(f"Processed {total_frames} frames. Current FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("Stopped.")

final_elapsed = time.time() - start_time
print(f"DONE. Average Inference FPS: {total_frames / final_elapsed:.2f}")

cap.release()