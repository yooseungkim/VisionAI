import cv2
import numpy as np
from ultralytics import YOLO

def apply_mosaic(image, x1, y1, x2, y2, ratio=25):
    """
    Applies mosaic with safety checks for image boundaries and minimum size.
    """
    # 1. Boundary Clamping (Ensure coordinates are within image bounds)
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    
    # Calculate ROI dimensions
    w = x2 - x1
    h = y2 - y1
    
    # 2. Safety Check: Skip if ROI is invalid or too small
    # If the ROI is smaller than the ratio, pixelation is mathematically impossible/meaningless via downsampling.
    if w < ratio or h < ratio:
        return image

    # Define ROI
    roi = image[y1:y2, x1:x2]
    
    try:
        # 3. Downsampling (Pixelate)
        # Using explicit max(1, ...) is a secondary safety, though the check above covers it.
        small_w = max(1, w // ratio)
        small_h = max(1, h // ratio)
        
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # 4. Upsampling
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Apply back
        image[y1:y2, x1:x2] = mosaic
        
    except Exception as e:
        # Fallback just in case specific resize ops fail
        print(f"Mosaic skipped for box {x1, y1, x2, y2}: {e}")
        return image

    return image

def process_video(input_path, output_path, model_path='yolov8n-face.pt'):
    # Load Model (Assume a model trained for face/license_plate exists)
    # Ideally, class 0: face, class 1: license_plate
    model = YOLO(model_path) 
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return

    # Video Writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing started...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inference
        results = model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                # Check specific classes (Example logic)
                # Replace these class IDs with your actual model's IDs
                # e.g., if you use a custom model where 0 is face, 1 is plate
                target_classes = [0, 1] 
                
                if cls in target_classes:
                    frame = apply_mosaic(frame, x1, y1, x2, y2, ratio=20)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done. Saved to {output_path}")

# Example Usage
process_video('/home/yooseung/workspace/VisionAI/tracking/datasets/videos/ch07_night.mp4', '/home/yooseung/workspace/VisionAI/tracking/datasets/videos/ch07_night_mosaic.mp4', 'yolo11n-seg.pt')