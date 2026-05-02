"""
Batch Video Analysis Script for Human Activity Recognition

This script processes multiple video files and generates a detailed
analysis report with activity predictions, confidence scores, and
timing information. Useful for batch processing and evaluation.

Usage:
    python batch_analyze.py --model resnet-34_kinetics.onnx --classes actions.txt \
        --input_dir videos/ --output_dir results/
"""

import os
import sys
import csv
import json
import time
import argparse
import numpy as np
import cv2
import imutils
from datetime import datetime


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Video Analysis for Human Activity Recognition"
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Path to pre-trained ONNX model")
    parser.add_argument("-c", "--classes", required=True,
                        help="Path to class labels file")
    parser.add_argument("-i", "--input_dir", required=True,
                        help="Directory containing input video files")
    parser.add_argument("-o", "--output_dir", default="results",
                        help="Directory to save analysis results")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="Use GPU/CUDA acceleration (1=yes, 0=no)")
    return vars(parser.parse_args())


def analyze_video(video_path, net, classes, sample_duration=16, sample_size=112):
    """
    Analyze a single video file for human activities.
    
    Args:
        video_path: Path to the video file
        net: Loaded OpenCV DNN network
        classes: List of class labels
        sample_duration: Number of frames per temporal window
        sample_size: Spatial size of input frames
    
    Returns:
        Dictionary containing analysis results
    """
    vs = cv2.VideoCapture(video_path)
    
    if not vs.isOpened():
        return {"error": f"Could not open video: {video_path}"}
    
    fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    predictions = []
    activity_counts = {}
    frame_idx = 0
    start_time = time.time()
    
    while True:
        frames = []
        
        for _ in range(sample_duration):
            grabbed, frame = vs.read()
            if not grabbed:
                break
            frame = imutils.resize(frame, width=400)
            frames.append(frame)
            frame_idx += 1
        
        if len(frames) < sample_duration:
            break
        
        # Create blob and run inference
        blob = cv2.dnn.blobFromImages(
            frames, 1.0,
            (sample_size, sample_size),
            (114.7748, 107.7354, 99.4750),
            swapRB=True, crop=True
        )
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)
        
        net.setInput(blob)
        outputs = net.forward()
        
        # Softmax for confidence
        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp_outputs / exp_outputs.sum()
        
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        label = classes[class_idx]
        
        # Get top-5 predictions
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(classes[i], float(probs[i])) for i in top5_idx]
        
        timestamp = frame_idx / fps if fps > 0 else 0
        
        predictions.append({
            "frame_start": frame_idx - sample_duration,
            "frame_end": frame_idx,
            "timestamp": round(timestamp, 2),
            "activity": label,
            "confidence": round(confidence, 4),
            "top5": top5
        })
        
        # Count activities
        activity_counts[label] = activity_counts.get(label, 0) + 1
    
    vs.release()
    processing_time = time.time() - start_time
    
    # Determine dominant activity
    dominant_activity = max(activity_counts, key=activity_counts.get) if activity_counts else "Unknown"
    
    return {
        "video_path": video_path,
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 1)
        },
        "analysis": {
            "frames_processed": frame_idx,
            "windows_analyzed": len(predictions),
            "processing_time_seconds": round(processing_time, 2),
            "dominant_activity": dominant_activity,
            "activity_distribution": activity_counts
        },
        "predictions": predictions
    }


def generate_report(results, output_dir):
    """Generate analysis reports in JSON and CSV formats."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_path = os.path.join(output_dir, f"analysis_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[INFO] JSON report saved: {json_path}")
    
    # CSV summary
    csv_path = os.path.join(output_dir, f"analysis_summary_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video", "Resolution", "Duration(s)", "Frames",
            "Dominant Activity", "Windows Analyzed", "Processing Time(s)"
        ])
        for result in results:
            if "error" not in result:
                writer.writerow([
                    os.path.basename(result["video_path"]),
                    result["video_info"]["resolution"],
                    result["video_info"]["duration_seconds"],
                    result["video_info"]["total_frames"],
                    result["analysis"]["dominant_activity"],
                    result["analysis"]["windows_analyzed"],
                    result["analysis"]["processing_time_seconds"]
                ])
    print(f"[INFO] CSV summary saved: {csv_path}")


def main():
    args = parse_args()
    
    # Load resources
    classes = open(args["classes"]).read().strip().split("\n")
    net = cv2.dnn.readNet(args["model"])
    
    if args["gpu"] > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    video_files = [
        os.path.join(args["input_dir"], f)
        for f in os.listdir(args["input_dir"])
        if os.path.splitext(f)[1].lower() in video_extensions
    ]
    
    if not video_files:
        print(f"[ERROR] No video files found in: {args['input_dir']}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(video_files)} video files to analyze")
    
    # Process each video
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Analyzing: {os.path.basename(video_path)}")
        result = analyze_video(video_path, net, classes)
        results.append(result)
        
        if "error" not in result:
            print(f"  → Dominant activity: {result['analysis']['dominant_activity']}")
            print(f"  → Windows analyzed: {result['analysis']['windows_analyzed']}")
            print(f"  → Processing time: {result['analysis']['processing_time_seconds']}s")
    
    # Generate reports
    generate_report(results, args["output_dir"])
    
    print(f"\n[DONE] Analyzed {len(results)} videos. Reports saved to: {args['output_dir']}")


if __name__ == "__main__":
    main()
