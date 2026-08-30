from ultralytics import YOLO
import cv2
import os
import sys

# Paths setup
model_path = './training/yolov8n/train/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'yolov8n.pt' # Fallback to local pretrained weights

video_path = './Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4'
if len(sys.argv) > 1:
    video_path = sys.argv[1]

if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
    print("Usage: python tracking.py <path_to_video.mp4>")
    print("Or use: python test_video.py --video <path_to_video.mp4>")
    sys.exit(1)

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

print(f"Processing video: {video_path}")
results = model.track(video_path, persist=True, stream=True, conf=0.25, task='detect', classes=[0])

cap = cv2.VideoCapture(video_path)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

output_path = "output_tracked.avi"
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

max_track_id = 0
unique_ids = set()

for result in results:
    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        for tid in track_ids:
            unique_ids.add(tid)
            if tid > max_track_id:
                max_track_id = tid
                
    tracked_frame = result.plot()
    output.write(tracked_frame)

output.release()
cap.release()
cv2.destroyAllWindows()

print("Tracking video complete...")
print(f"Output saved to: {output_path}")
print(f"There are {len(unique_ids)} unique people detected/tracked in video (Max ID: {max_track_id})")