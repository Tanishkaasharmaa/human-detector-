import argparse
import os
import sys
import cv2
from ultralytics import YOLO

def find_default_model():
    """Locate the best trained model or fall back to pretrained YOLOv8 weights."""
    possible_paths = [
        './training/yolov8n/train/weights/best.pt',
        './training/yolov8s/train/weights/best.pt',
        './training/yolov8m/train/weights/best.pt',
        'yolov8n.pt',
        'yolov8s.pt',
        'yolov8m.pt'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return 'yolov8n.pt'

def main():
    parser = argparse.ArgumentParser(description="Test and Track Persons in Video using YOLOv8 Model")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (default: auto-detect best trained or pretrained model)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (e.g., output_tracked.mp4 or output_tracked.avi)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracking config: 'bytetrack.yaml' or 'botsort.yaml'")
    parser.add_argument("--percent", type=float, default=100.0, help="Percentage of video to process (e.g., 20 for 20%)")
    parser.add_argument("--show", action="store_true", help="Display live window during processing")
    
    args = parser.parse_args()

    # 1. Validate video input
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        sys.exit(1)

    # 2. Resolve model path
    model_path = args.model if args.model else find_default_model()
    if not os.path.exists(model_path) and not model_path.endswith('.pt'):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    # 3. Open video capture & frame metadata
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{args.video}'.")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle invalid FPS
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_frames = total_frames
    if 0 < args.percent < 100.0 and total_frames > 0:
        target_frames = max(1, int(total_frames * (args.percent / 100.0)))

    print("==================================================")
    print(f"Input Video    : {args.video}")
    print(f"Model Path     : {model_path}")
    print(f"Confidence     : {args.conf}")
    print(f"Video Res/FPS  : {width}x{height} @ {fps:.2f} FPS")
    print(f"Total Frames   : {total_frames}")
    if target_frames != total_frames:
        print(f"Processing Cap : {target_frames} frames ({args.percent}% of video)")
    print("==================================================")

    # 4. Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)

    # 5. Configure output video writer
    if not args.output:
        base, ext = os.path.splitext(os.path.basename(args.video))
        output_path = f"{base}_tracked.mp4"
    else:
        output_path = args.output

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 6. Run tracking generator
    print("Starting person detection and tracking...")
    results = model.track(
        source=args.video,
        persist=True,
        stream=True,
        conf=args.conf,
        tracker=args.tracker,
        classes=[0]  # Class 0 in COCO is 'person'
    )

    max_track_id = 0
    unique_ids = set()
    frame_count = 0

    for result in results:
        frame_count += 1
        
        # Track IDs
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            for tid in track_ids:
                unique_ids.add(tid)
                if tid > max_track_id:
                    max_track_id = tid

        # Plot result frame
        tracked_frame = result.plot()

        # Write frame to output video
        out.write(tracked_frame)

        # Optional display
        if args.show:
            try:
                cv2.imshow("Person Tracking", tracked_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user.")
                    break
            except Exception as e:
                print("\nWarning: Live preview window is not supported by your installed OpenCV package.")
                print("Continuing video processing in background mode...\n")
                args.show = False

        if frame_count % 30 == 0 or frame_count == target_frames:
            print(f"Frame {frame_count}/{target_frames} processed | Unique Persons Tracked: {len(unique_ids)}")

        if target_frames > 0 and frame_count >= target_frames:
            print(f"\nReached requested limit of {target_frames} frames ({args.percent}% of video). Stopping tracking.")
            break

    # Clean up
    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    print("==================================================")
    print("Tracking complete!")
    print(f"Output Video Saved To : {os.path.abspath(output_path)}")
    print(f"Total Unique Persons Tracked : {len(unique_ids)}")
    print("==================================================")

if __name__ == "__main__":
    main()
