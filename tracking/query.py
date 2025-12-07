import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. Configuration (Local SOTA Environment)
# ==========================================
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

def setup_model():
    """RTX 4090에 최적화된 Qwen2-VL 모델을 로드합니다."""
    print(f"[System] Loading {MODEL_ID} to GPU with Flash Attention 2...")
    
    # 4-bit Quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # min/max pixels control token usage. 
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1024*28*28)
    return model, processor

# ==========================================
# 2. Data Processing Utils
# ==========================================

def load_prompt_from_file(file_path):
    """외부 텍스트 파일에서 프롬프트를 읽어옵니다."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_raw_logs(file_path):
    """track.py에서 생성된 jsonl 로그를 읽어옵니다."""
    events = []
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                continue
    return events

def merge_close_events(events, time_threshold=3.0):
    """
    시간적으로 인접한(3초 이내) 이벤트들을 하나로 병합합니다.
    데이터가 문자열로 들어올 경우를 대비해 float() 형변환을 수행합니다.
    """
    if not events:
        return []

    # Helper function to safely get float value
    def get_float(item, key, default=0.0):
        try:
            val = item.get(key, default)
            return float(val)
        except (ValueError, TypeError):
            return float(default)

    # timestamp 기준 정렬
    events.sort(key=lambda x: get_float(x, 'timestamp'))

    merged = []
    current_event = events[0]

    for next_event in events[1:]:
        # 형변환 적용하여 계산
        curr_start = get_float(current_event, 'timestamp')
        curr_dur = get_float(current_event, 'duration')
        curr_end = curr_start + curr_dur
        
        next_start = get_float(next_event, 'timestamp')
        next_dur = get_float(next_event, 'duration')

        # 시간 차이 계산
        if next_start - curr_end <= time_threshold:
            # 병합 로직: 종료 시간 연장
            new_end_time = max(curr_end, next_start + next_dur)
            new_duration = new_end_time - curr_start
            
            # 값 업데이트 (일관성을 위해 문자열 포맷팅 or float 유지)
            current_event['duration'] = new_duration 
            # 필요시 actors 리스트 병합 등 추가 로직 가능
        else:
            merged.append(current_event)
            current_event = next_event

    merged.append(current_event)
    return merged

def get_frames(vr, start_time, duration, num_frames=16):
    """decord를 사용하여 특정 구간의 프레임을 추출합니다."""
    fps = vr.get_avg_fps()
    end_time = start_time + duration
    total_frames = len(vr)
    
    start_idx = int(start_time * fps)
    end_idx = int(end_time * fps)
    
    start_idx = max(0, start_idx)
    end_idx = min(total_frames - 1, end_idx)
    
    if start_idx >= end_idx:
        indices = [start_idx]
    else:
        indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=int)
        
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]

# ==========================================
# 3. Main Inference Logic
# ==========================================

def analyze_events_with_qwen(model, processor, video_path, events):
    """
    병합된 이벤트를 순회하며 비디오 프레임을 추출하고 Qwen2-VL로 분석합니다.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    refined_results = []
    
    print(f"[Analysis] Starting analysis on {len(events)} merged events...")

    for i, event in enumerate(events):
        start_t = event.get('timestamp', 0)
        duration = event.get('duration', 1)
        
        print(f"  > Processing Event {i+1}: {start_t:.2f}s (Dur: {duration:.2f}s)")
        
        # 1. Extract Frames
        frames = get_frames(vr, start_t, duration)
        
        # 2. Build Prompt (JSON Output 강제)
        prompt = load_prompt_from_file("./prompt_eng.txt")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                        "fps": 25, 
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # 3. Inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True
            )
            
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 4. Parse Output (Text -> JSON Structure)
        # Qwen output usually contains the prompt, so we strip it
        response_part = output_text.split("assistant\n")[-1].strip()
        
        # Markdown backticks 제거
        clean_json_str = response_part.replace("```json", "").replace("```", "").strip()
        
        try:
            analysis_data = json.loads(clean_json_str)
        except json.JSONDecodeError:
            # Fallback if model fails to output strict JSON
            analysis_data = {"summary": clean_json_str, "actors": [], "risk_level": "unknown"}

        # 5. Combine with original metadata
        refined_event = {
            "frame": int(start_t * vr.get_avg_fps()),
            "timestamp": f"{start_t:.2f}",
            "duration": f"{duration:.2f}",
            "event": analysis_data.get("summary", "No description"),
            "warning": analysis_data.get("risk_level", "low"),
            "actors": analysis_data.get("actors", []),
            "raw_log_reference": event  # 원본 로그 백업
        }
        refined_results.append(refined_event)

    return refined_results

def save_jsonl(events, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

# ==========================================
# 4. Entry Point
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to the source video file')
    parser.add_argument('--input', type=str, required=True, help='Path to raw logs from track.py')
    parser.add_argument('--output', type=str, required=True, help='Path to save refined prediction')
    args = parser.parse_args()

    # 1. Load Data
    print(f"[Query] Loading raw logs from: {args.input}")
    raw_data = load_raw_logs(args.input)
    
    # 2. Preprocess (Merge)
    merged_data = merge_close_events(raw_data)
    print(f"[Query] Merged {len(raw_data)} raw events into {len(merged_data)} distinct segments.")
    
    # 3. Model Setup & Inference
    model, processor = setup_model()
    refined_data = analyze_events_with_qwen(model, processor, args.video, merged_data)
    
    # 4. Save
    print(f"[Query] Saving {len(refined_data)} refined events to: {args.output}")
    save_jsonl(refined_data, args.output)