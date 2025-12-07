import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

# ==========================================
# Pipeline Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATASET_VIDEO_DIR = os.path.join(BASE_DIR, "datasets", "videos")
DATASET_GT_DIR    = os.path.join(BASE_DIR, "datasets", "logs")

RESULT_LOG_DIR    = os.path.join(BASE_DIR, "results", "logs")
RESULT_PRED_DIR   = os.path.join(BASE_DIR, "results", "preds")
RESULT_EVAL_DIR   = os.path.join(BASE_DIR, "results", "evals")

# Scripts
SCRIPT_TRACK = os.path.join(BASE_DIR, "track.py")
SCRIPT_QUERY = os.path.join(BASE_DIR, "query.py")
SCRIPT_EVAL  = os.path.join(BASE_DIR, "eval.py")

def run_command(command, description):
    s = time.time()
    """Helper function to run subprocess"""
    print(f"\n{'='*60}")
    print(f"🚀 [Step: {description}]")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, text=True)
        print(f"Executed for {time.time() - s:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        return False

def process_video(video_file, overwrite=False):
    video_name = video_file.name
    base_name = video_file.stem
    
    print(f"\n\n>>> Processing Video: {video_name} <<<")

    # -------------------------------------------------
    # 1. Tracking (track.py)
    # Input: Video -> Output: Raw JSONL Log
    # -------------------------------------------------
    raw_log_path = os.path.join(RESULT_LOG_DIR, f"events_{base_name}.jsonl")
    
    # Check if raw log exists
    if os.path.exists(raw_log_path) and not overwrite:
        print(f"ℹ️ [Skip] Tracking log already exists: {raw_log_path}")
    else:
        cmd_track = [sys.executable, SCRIPT_TRACK, "--source", str(video_file)]
        if not run_command(cmd_track, "Tracking & Detection"):
            return

    # Tracking 결과 확인 (Skip 했더라도 다음 단계를 위해 파일 존재 여부 재확인)
    if not os.path.exists(raw_log_path):
        print(f"⚠️ Warning: Log file not found at {raw_log_path}. Cannot proceed to Reasoning.")
        return

    # -------------------------------------------------
    # 2. Reasoning (query.py)
    # Input: Raw JSONL -> Output: Refined Pred JSONL
    # -------------------------------------------------
    pred_log_path = os.path.join(RESULT_PRED_DIR, f"events_{base_name}.jsonl")
    
    # Check if prediction log exists
    if os.path.exists(pred_log_path) and not overwrite:
        print(f"ℹ️ [Skip] Prediction log already exists: {pred_log_path}")
    else:
        cmd_query = [
            sys.executable, SCRIPT_QUERY,
            "--video", os.path.join(DATASET_VIDEO_DIR,video_file.name), 
            "--input", raw_log_path, 
            "--output", pred_log_path
        ]
        if not run_command(cmd_query, "LLM Reasoning (Gemini)"):
            return

    # Prediction 결과 확인
    if not os.path.exists(pred_log_path):
        print(f"⚠️ Warning: Prediction file not found at {pred_log_path}. Cannot proceed to Evaluation.")
        return

    # -------------------------------------------------
    # 3. Evaluation (eval.py)
    # Input: GT JSONL & Pred JSONL -> Output: Eval Report
    # -------------------------------------------------
    gt_log_path = os.path.join(DATASET_GT_DIR, f"events_{base_name}.jsonl")
    
    # Evaluation은 실행 시간이 짧으므로 보통 매번 실행하지만, 
    # 필요하다면 여기에만 overwrite 로직을 적용하지 않거나 별도로 처리 가능합니다.
    # 현재 로직: GT와 Prediction이 모두 있을 때 수행
    if os.path.exists(gt_log_path):
        eval_output_path = os.path.join(RESULT_EVAL_DIR, f"report_{base_name}.txt")
        os.makedirs(RESULT_EVAL_DIR, exist_ok=True)
        
        # Report가 이미 있고 overwrite가 꺼져있으면 Skip 할 수도 있으나,
        # 보통 평가는 다시 보고 싶을 수 있으므로 여기서는 수행하도록 둠 (원하면 조건문 추가 가능)
        
        with open(eval_output_path, "w") as outfile:
            cmd_eval = [
                sys.executable, SCRIPT_EVAL,
                "--gt", gt_log_path,
                "--pred", pred_log_path
            ]
            
            print(f"\n{'='*60}")
            print(f"🚀 [Step: Evaluation]")
            print(f"Saving report to: {eval_output_path}")
            print(f"{'='*60}")
            
            subprocess.run(cmd_eval, stdout=outfile, text=True)
            
            print("\n[Evaluation Summary]")
            with open(eval_output_path, "r") as f:
                print(f.read())
    else:
        print(f"ℹ️ Skipping Evaluation: No GT found at {gt_log_path}")

def main():
    parser = argparse.ArgumentParser(description="Parking Surveillance Pipeline")
    parser.add_argument("--video", type=str, help="Specific video name (e.g., parking7.mp4). If empty, runs all.")
    parser.add_argument("--overwrite", action="store_true", help="Force re-run even if output files exist.")
    args = parser.parse_args()

    # Create directories
    for path in [DATASET_VIDEO_DIR, DATASET_GT_DIR, RESULT_LOG_DIR, RESULT_PRED_DIR, RESULT_EVAL_DIR]:
        os.makedirs(path, exist_ok=True)

    # Search videos
    video_files = []
    if args.video:
        target = Path(DATASET_VIDEO_DIR) / args.video
        if target.exists():
            video_files.append(target)
        else:
            print(f"❌ Video not found: {target}")
    else:
        video_files = list(Path(DATASET_VIDEO_DIR).glob("*.mp4"))

    if not video_files:
        print(f"❌ No videos found in {DATASET_VIDEO_DIR}")
        return

    print(f"Found {len(video_files)} videos to process.")
    
    for video in video_files:
        process_video(video, overwrite=args.overwrite)

if __name__ == "__main__":
    main()