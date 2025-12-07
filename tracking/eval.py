import argparse
import json
import sys
import os

def calculate_iou(range1, range2):
    """
    Calculate temporal Intersection over Union (t-IoU)
    range: (start_frame, end_frame)
    """
    start1, end1 = range1
    start2, end2 = range2
    
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    if intersection_end <= intersection_start:
        return 0.0
        
    intersection = intersection_end - intersection_start
    union = max(end1, end2) - min(start1, start2)
    
    if union == 0: return 0.0
    
    return intersection / union

def load_file(file_path):
    """
    Robustly loads JSON or JSONL files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        sys.exit(1)

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try reading line by line (JSONL)
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Fallback: Try reading the whole file as a standard JSON array
                try:
                    f.seek(0)
                    return json.load(f)
                except:
                    print(f"Error: Failed to parse line {line_num+1} in {file_path}")
                    continue
    return data

def evaluate(gt_path, pred_path):
    # 1. Load Data
    gt_data = load_file(gt_path)
    pred_data = load_file(pred_path)

    print(f"\nEvaluating: {os.path.basename(pred_path)} against {os.path.basename(gt_path)}")
    print(f"Loaded {len(gt_data)} GT events and {len(pred_data)} Pred events.")
    print("=" * 85)
    print(f"{'Event Type':<25} | {'Range (Frames)':<15} | {'t-IoU':<8} | {'Risk Diff':<10}")
    print("-" * 85)

    total_iou = 0
    total_risk_diff = 0
    matched_count = 0
    risk_eval_count = 0
    
    # 2. Evaluation Loop (Iterate over GT)
    for gt_item in gt_data:
        event_type = gt_item.get('event', 'unknown')
        
        # New JSON format parsing
        gt_start = gt_item.get('start_frame')
        gt_end = gt_item.get('end_frame')
        
        # Skip invalid data
        if gt_start is None or gt_end is None:
            continue
            
        gt_range = (gt_start, gt_end)
        gt_risk = gt_item.get('warning') # Can be None if dataset doesn't have it

        best_iou = 0.0
        best_pred = None
        
        # Greedy Matching: Find best overlapping prediction with same label
        for pred_item in pred_data:
            if pred_item.get('event') != event_type:
                continue
            
            # Use same keys for prediction
            p_start = pred_item.get('start_frame')
            p_end = pred_item.get('end_frame')
            
            if p_start is None or p_end is None:
                continue
                
            pred_range = (p_start, p_end)
            iou = calculate_iou(gt_range, pred_range)
            
            if iou > best_iou: 
                best_iou = iou
                best_pred = pred_item

        # 3. Calculate Metrics for this event
        if best_pred:
            pred_risk = best_pred.get('warning')
            
            # Risk Diff Calculation (Only if both exist)
            risk_diff_str = "-"
            if gt_risk is not None and pred_risk is not None:
                diff_val = abs(float(gt_risk) - float(pred_risk))
                risk_diff_str = f"{diff_val:.1f}"
                total_risk_diff += diff_val
                risk_eval_count += 1
            
            total_iou += best_iou
            matched_count += 1
            
            # Display format: "100-200"
            range_str = f"{gt_start}-{gt_end}"
            print(f"{event_type:<25} | {range_str:<15} | {best_iou:.2f}{'':<4} | {risk_diff_str}")
        else:
            # No match found (False Negative)
            range_str = f"{gt_start}-{gt_end}"
            print(f"{event_type:<25} | {range_str:<15} | {'0.00 (Miss)':<12} | -")
            
    # 4. Final Summary
    print("=" * 85)
    print(" [ Final Report ]")
    
    final_iou = total_iou / len(gt_data) if len(gt_data) > 0 else 0
    # Recall based IoU (IoU sum / Total GT) to penalize misses
    
    print(f" -> Average t-IoU (Recall) : {final_iou:.4f}")
    
    if risk_eval_count > 0:
        final_mae = total_risk_diff / risk_eval_count
        print(f" -> Risk MAE               : {final_mae:.4f}")
    else:
        print(" -> Risk MAE               : N/A (Risk scores not found in data)")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True, help='Path to GT file (.jsonl)')
    parser.add_argument('--pred', type=str, required=True, help='Path to Prediction file (.jsonl)')
    args = parser.parse_args()
    
    evaluate(args.gt, args.pred)