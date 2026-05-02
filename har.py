"""
Human Activity Recognition (HAR) using OpenCV and Deep Learning

This module implements real-time human activity recognition using a pre-trained
ResNet-34 3D Convolutional Neural Network trained on the Kinetics-400 dataset.
It processes video frames in temporal windows and classifies activities from
400+ categories including sports, daily activities, and gestures.

Architecture:
    - Model: ResNet-34 (3D CNN) exported to ONNX format
    - Dataset: Kinetics-400 (400 human action categories)
    - Input: 16-frame temporal window, each frame resized to 112x112
    - Output: Predicted activity label with confidence score

Usage:
    Webcam:
        python har.py --model resnet-34_kinetics.onnx --classes actions.txt
    
    Video file:
        python har.py --model resnet-34_kinetics.onnx --classes actions.txt \
            --input videos/example.mp4 --output output.mp4
    
    With GPU acceleration:
        python har.py --model resnet-34_kinetics.onnx --classes actions.txt --gpu 1

Author: Manish
Project: Human Activity Recognition for AI/ML Portfolio
"""

import numpy as np
import argparse
import imutils
import sys
import cv2
import time
import os


def parse_arguments():
    """Parse command-line arguments for the HAR system."""
    parser = argparse.ArgumentParser(
        description="Human Activity Recognition using Deep Learning & OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Webcam mode:
    python har.py --model resnet-34_kinetics.onnx --classes actions.txt
  
  Video file mode:
    python har.py --model resnet-34_kinetics.onnx --classes actions.txt \\
        --input videos/example.mp4 --output output.mp4
  
  With GPU:
    python har.py --model resnet-34_kinetics.onnx --classes actions.txt --gpu 1
        """
    )
    parser.add_argument("-m", "--model", required=True,
                        help="Path to pre-trained ResNet-34 ONNX model")
    parser.add_argument("-c", "--classes", required=True,
                        help="Path to class labels text file (Kinetics-400)")
    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to input video file (leave empty for webcam)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Path to output video file")
    parser.add_argument("-d", "--display", type=int, default=1,
                        help="Whether to display output frame (1=yes, 0=no)")
    parser.add_argument("-g", "--gpu", type=int, default=0,
                        help="Whether to use GPU/CUDA acceleration (1=yes, 0=no)")
    return vars(parser.parse_args())


def load_class_labels(filepath):
    """
    Load activity class labels from a text file.
    
    Args:
        filepath: Path to the text file containing class labels (one per line)
    
    Returns:
        List of activity class label strings
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Class labels file not found: {filepath}")
        sys.exit(1)
    
    classes = open(filepath).read().strip().split("\n")
    print(f"[INFO] Loaded {len(classes)} activity classes")
    return classes


def load_model(model_path, use_gpu=False):
    """
    Load the pre-trained deep learning model for activity recognition.
    
    Args:
        model_path: Path to the ONNX model file
        use_gpu: Whether to use CUDA GPU acceleration
    
    Returns:
        OpenCV DNN network object
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("[INFO] Please download the ResNet-34 Kinetics ONNX model.")
        print("[INFO] Run: python download_model.py")
        sys.exit(1)
    
    print("[INFO] Loading deep learning model for Human Activity Recognition...")
    net = cv2.dnn.readNet(model_path)
    
    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    else:
        print("[INFO] Using CPU for inference")
    
    print("[INFO] Model loaded successfully!")
    return net


def create_blob(frames, sample_size=112):
    """
    Create a Binary Large Object (blob) from a sequence of frames
    for input to the neural network.
    
    The blob transformation includes:
    1. Resizing frames to sample_size x sample_size
    2. Mean subtraction for normalization (ImageNet means)
    3. Channel swapping (BGR to RGB)
    4. Reshaping to 5D tensor: [batch, channels, depth, height, width]
    
    Args:
        frames: List of video frames (numpy arrays)
        sample_size: Target spatial dimension (default: 112x112)
    
    Returns:
        5D numpy array blob ready for network input
    """
    blob = cv2.dnn.blobFromImages(
        frames, 
        1.0,
        (sample_size, sample_size),
        (114.7748, 107.7354, 99.4750),  # Mean values for normalization
        swapRB=True, 
        crop=True
    )
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    return blob


def draw_prediction(frame, label, confidence, fps_text=""):
    """
    Draw the prediction overlay on a video frame with a styled HUD.
    
    Args:
        frame: Video frame to draw on
        label: Predicted activity label
        confidence: Prediction confidence (0-1)
        fps_text: FPS counter string
    
    Returns:
        Frame with prediction overlay drawn
    """
    h, w = frame.shape[:2]
    
    # --- Main prediction banner (top) ---
    overlay = frame.copy()
    # Semi-transparent dark banner
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Activity label
    label_text = f"Activity: {label}"
    cv2.putText(frame, label_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    
    # Confidence bar
    conf_pct = confidence * 100
    bar_width = int(250 * confidence)
    
    # Bar background
    cv2.rectangle(frame, (15, 45), (265, 60), (50, 50, 50), -1)
    # Confidence fill (green gradient based on confidence)
    color = (0, int(255 * confidence), int(255 * (1 - confidence)))
    cv2.rectangle(frame, (15, 45), (15 + bar_width, 60), color, -1)
    # Confidence text
    cv2.putText(frame, f"{conf_pct:.1f}%", (275, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # FPS counter (top-right)
    if fps_text:
        cv2.putText(frame, fps_text, (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    
    # --- Bottom info bar ---
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "HAR System | ResNet-34 | Kinetics-400 | Press 'q' to quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    return frame


def main():
    """Main function to run the Human Activity Recognition system."""
    # Parse arguments
    args = parse_arguments()
    
    # Configuration constants
    SAMPLE_DURATION = 16   # Number of frames per temporal window
    SAMPLE_SIZE = 112      # Spatial size of each frame for the model
    
    # Load class labels and model
    CLASSES = load_class_labels(args["classes"])
    net = load_model(args["model"], use_gpu=(args["gpu"] > 0))
    
    # Initialize video stream
    source = args["input"] if args["input"] else 0
    source_name = args["input"] if args["input"] else "Webcam"
    print(f"[INFO] Accessing video stream: {source_name}")
    
    vs = cv2.VideoCapture(source)
    
    if not vs.isOpened():
        print(f"[ERROR] Could not open video source: {source_name}")
        sys.exit(1)
    
    # Get video properties
    fps = vs.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    if total_frames > 0:
        duration = total_frames / fps
        print(f"[INFO] Total frames: {total_frames} | Duration: {duration:.1f}s")
    
    # Initialize output writer
    writer = None
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # We'll initialize the writer once we know the output frame size
    
    # Performance tracking
    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    print("\n" + "=" * 60)
    print("  HUMAN ACTIVITY RECOGNITION SYSTEM - RUNNING")
    print("  Press 'q' to quit")
    print("=" * 60 + "\n")
    
    # Main processing loop
    try:
        while True:
            frames = []       # Resized frames for model input
            originals = []    # Original frames for display
            
            # Collect SAMPLE_DURATION frames (temporal window)
            for i in range(SAMPLE_DURATION):
                grabbed, frame = vs.read()
                
                if not grabbed:
                    if args["input"]:
                        print("\n[INFO] End of video stream reached.")
                    else:
                        print("\n[INFO] No frame read from webcam - exiting...")
                    
                    # Write any remaining frames and clean up
                    if writer is not None:
                        writer.release()
                    vs.release()
                    cv2.destroyAllWindows()
                    
                    # Print summary
                    elapsed = time.time() - start_time
                    print(f"\n[SUMMARY] Processed {frame_count} frames in {elapsed:.1f}s")
                    if inference_times:
                        avg_inference = np.mean(inference_times) * 1000
                        print(f"[SUMMARY] Average inference time: {avg_inference:.1f}ms")
                    sys.exit(0)
                
                originals.append(frame)
                resized = imutils.resize(frame, width=400)
                frames.append(resized)
            
            # Create blob from the frame window
            blob = create_blob(frames, SAMPLE_SIZE)
            
            # Run inference
            inference_start = time.time()
            net.setInput(blob)
            outputs = net.forward()
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            # Get prediction
            class_idx = np.argmax(outputs)
            confidence = float(outputs[0][class_idx])
            
            # Apply softmax for proper probability
            exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
            softmax_probs = exp_outputs / exp_outputs.sum()
            confidence = float(softmax_probs[class_idx])
            
            label = CLASSES[class_idx]
            
            # Calculate FPS
            current_fps = SAMPLE_DURATION / (time.time() - start_time) * (frame_count / SAMPLE_DURATION + 1) if frame_count > 0 else 0
            processing_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_text = f"FPS: {processing_fps:.1f}"
            
            # Log prediction
            frame_count += SAMPLE_DURATION
            print(f"[PREDICTION] Frame {frame_count:>6d} | "
                  f"Activity: {label:<35s} | "
                  f"Confidence: {confidence*100:>5.1f}% | "
                  f"Inference: {inference_time*1000:.0f}ms")
            
            # Render predictions on each frame in the window
            for frame in originals:
                # Draw styled prediction overlay
                frame = draw_prediction(frame, label, confidence, fps_text)
                
                # Display the frame
                if args["display"] > 0:
                    cv2.imshow("Human Activity Recognition", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt
                
                # Initialize writer if needed (first time)
                if args["output"] and writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(
                        args["output"], fourcc, fps,
                        (frame.shape[1], frame.shape[0]), True
                    )
                    print(f"[INFO] Writing output to: {args['output']}")
                
                # Write frame to output
                if writer is not None:
                    writer.write(frame)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user - cleaning up...")
    
    finally:
        # Cleanup
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"  SESSION SUMMARY")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total time: {elapsed:.1f}s")
        if inference_times:
            avg_inference = np.mean(inference_times) * 1000
            print(f"  Average inference time: {avg_inference:.1f}ms per batch")
            print(f"  Average throughput: {frame_count / elapsed:.1f} frames/s")
        print(f"{'=' * 60}\n")
        
        if writer is not None:
            writer.release()
            print(f"[INFO] Output video saved to: {args['output']}")
        
        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
