import json
import os
import subprocess
import re
import argparse
from pathlib import Path

def parse_time_str_to_seconds(time_str):
    """
    Converts string format "Xm Ys" (e.g., "19m 7s") to total seconds (float).
    """
    # Regex to find minutes and seconds
    match = re.search(r'(\d+)m\s*(\d+)s', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return float(minutes * 60 + seconds)
    
    # Fallback: if format is different (e.g. just numbers)
    try:
        return float(time_str)
    except ValueError:
        print(f"[Error] Cannot parse timestamp: {time_str}")
        return 0.0

def main(video_path, json_path, output_dir):
    # 1. Setup paths
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[Error] Video file not found: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    original_name = video_path.stem  # File name without extension

    # 2. Read JSON Lines file
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Start processing {len(lines)} events from {json_path}...")

    count = 0
    for line in lines:
        if not line.strip(): continue # Skip empty lines

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Warning] Skipped invalid JSON line: {e}")
            continue

        # 3. Extract Meta Data
        event_name = record.get("event", "UnknownEvent")
        timestamp_str = record.get("timestamp", "0m 0s")
        duration = record.get("duration_sec", 5.0) # Default 5s if missing
        
        # Risk Logic: Get 'risk_level' from 'risk_analysis', default to 0 if empty/missing
        risk_analysis = record.get("risk_analysis", {})
        risk_level = risk_analysis.get("risk_level", 0)

        # 4. Calculate Start Time (Seconds)
        start_seconds = parse_time_str_to_seconds(timestamp_str)

        # 5. Construct Output Filename
        # Requirement: {Original}_{Event}_{Timestamp}_{Risk}.mp4
        # Sanitize timestamp string for filename (remove spaces)
        safe_time_str = timestamp_str.replace(" ", "")
        output_filename = f"{original_name}_{event_name}_{safe_time_str}_{risk_level}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # 6. Run FFmpeg
        # -ss: Start time
        # -t: Duration (Easier than calculating End time with -to)
        # -c copy: Fast stream copy
        cmd = [
            "ffmpeg",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
            "-c", "copy",
            "-y",
            "-loglevel", "error",
            output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            count += 1
            if count % 5 == 0:
                print(f"Processed {count} clips...")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Failed to clip {output_filename}: {e}")

    print(f"Done! Total {count} clips saved in '{output_dir}'.")

    cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", "select='not(mod(n,25))',setpts=N/FRAME_RATE/TB",
            "-an",
            "-preset", "veryfast",
            "-r", "5", 
            os.path.join(output_dir, f"timelaps_{original_name}.mp4")
        ]

    try:
        subprocess.run(cmd, check=True)
        count += 1
        if count % 5 == 0:
            print(f"Processed Timelaps")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to generate timelaps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip video based on JSON Lines metadata.")
    
    
    # CLI Arguments
    parser.add_argument("--video", type=str, required=True, help="Path to the source video file")
    parser.add_argument("--json", type=str, required=True, help="Path to the metadata JSON file")
    parser.add_argument("--out", type=str, default="clips", help="Output directory (default: clips)")

    args = parser.parse_args()
    
    main(args.video, args.json, args.out)