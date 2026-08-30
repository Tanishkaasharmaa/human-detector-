# Person Tracking Project

## Introduction
This project focuses on tracking people in video frames using the YOLOv8 object detection model. The code provides three main components: exploratory data analysis (EDA) on the COCO 2017 dataset, training the YOLOv8 model on the filtered dataset, and then using the trained model to track people in a video.

## Features
- Performed EDA on the COCO 2017 dataset to filter for the 'person' class.
- Trained YOLOv8 models (yolov8n, yolov8s, yolov8m) on the filtered COCO 2017 dataset.
- Implemented video tracking using the trained YOLOv8 models to detect and track people in a video.
- Saved the tracked video with bounding boxes and unique IDs for each person.
- Reported the total number of people detected in the video.

## Prerequisites
- Python 3.x
- Ultralytics YOLO library
- FiftyOne library
- OpenCV library
- Matplotlib library

## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/person-tracking.git
```
2. Navigate to the project directory:
```
cd person-tracking
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Exploratory Data Analysis
Before training the YOLOv8 models, we performed an exploratory data analysis (EDA) on the COCO 2017 dataset to prepare the data for training.

1. Loaded the COCO 2017 dataset using the FiftyOne library, focusing on the 'person' class.
2. Filtered the dataset to only include samples with 'person' detections in the ground truth.
3. Exported the filtered dataset in the YOLOv5 format, with the 'person' class as the only label, to the `./yolov5-coco-datasets` directory.

This process ensured that the training, validation, and test splits of the dataset only contained samples with 'person' detections, which was the focus of our person-tracking project.

### Sample Data

![image](https://github.com/insomnius/person-detection/assets/20650401/366d8415-0cf0-4e2c-bfbb-05c611e4ec5a)

![image](https://github.com/insomnius/person-detection/assets/20650401/fa1a7b04-3cb1-40c1-86eb-2d0c1e3fc27d)

![image](https://github.com/insomnius/person-detection/assets/20650401/d62f425f-ec3d-424d-ae32-09b1d254cddc)

## Usage

### 1. Exploratory Data Analysis (EDA)
Load COCO 2017 dataset, filter for the 'person' class, and export in YOLOv5 format:
```bash
python eda.py
```

### 2. Model Training
Train YOLOv8 models (`yolov8n`, `yolov8s`, `yolov8m`) on the filtered dataset:
```bash
python train.py
```

### 3. Testing Video & Tracking
Use `test_video.py` to run person detection and tracking on any custom video:

- **Run on a full video**:
  ```bash
  python test_video.py --video "your_video.mp4"
  ```

- **Run on a percentage of the video** (e.g. process only 20% for quick testing):
  ```bash
  python test_video.py --video "your_video.mp4" --percent 20
  ```

- **Specify model weights & confidence threshold**:
  ```bash
  python test_video.py --video "your_video.mp4" --model "./training/yolov8n/train/weights/best.pt" --conf 0.3
  ```

- **Enable live window preview** (if supported by your OpenCV environment):
  ```bash
  python test_video.py --video "your_video.mp4" --show
  ```

---

## ⚡ Deployment on Edge Devices (Jetson Nano / Raspberry Pi)

This project can be deployed on embedded edge devices like **NVIDIA Jetson Nano** or **Raspberry Pi (Pi 4 / Pi 5)**. For optimal real-time performance (high FPS), export the PyTorch model to hardware-accelerated formats:

### 1. NVIDIA Jetson Nano Dev Kit (TensorRT)
Jetson Nano hardware is accelerated using TensorRT (`.engine`):

1. Export trained YOLOv8 model to TensorRT FP16:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="engine", half=True, device=0)
   ```
2. Run video tracking using TensorRT backend (~25–35+ FPS):
   ```bash
   python test_video.py --video "your_video.mp4" --model yolov8n.engine
   ```

### 2. Raspberry Pi 4 / Pi 5 (NCNN / TFLite)
Raspberry Pi ARM CPUs run fastest with NCNN or TFLite INT8 format:

1. Export trained YOLOv8 model to NCNN format:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="ncnn")
   ```
2. Run tracking on Raspberry Pi (~15–22 FPS on Pi 5):
   ```bash
   python test_video.py --video "your_video.mp4" --model yolov8n_ncnn_model
   ```

### 3. Live Webcam Stream on Edge Devices
To run real-time tracking from a USB webcam or CSI camera on Jetson Nano / Raspberry Pi:
```bash
python test_video.py --video 0
```

---

## Results

The trained YOLOv8 models achieved the following mean average precisions (mAP) on the COCO 2017 validation set:
- yolov8n: `0.61287`
- yolov8s: `0.56026`
- yolov8m: `0.59617`

### Detection

![image](https://github.com/insomnius/person-detection/assets/20650401/42914af4-b2c8-4de9-867d-fbf7b8b438d5)

### Tracking

The tracking code detects and tracks people with unique IDs across video frames.

<details open="" class="details-reset border rounded-2">
  <video src="https://github.com/insomnius/person-detection/assets/20650401/00f50d3a-13a8-4fdb-a4e1-f6cd5194224f" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">
  </video>
</details>

<br>

<details open="" class="details-reset border rounded-2">
  <video src="https://github.com/insomnius/person-detection/assets/20650401/7f7ad14d-3566-4503-949a-784cf1b7ef49" controls="controls" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="max-height:640px; min-height: 200px">
  </video>
</details>

## Future Improvements
- Integrate more advanced tracking algorithms to improve the accuracy and robustness of person tracking.
- Explore the use of other object detection models, such as YOLOv5 or Faster R-CNN, and compare performance.
- Implement real-time person tracking on live video streams.
- Explore integration of the person tracking system with other applications (people counting or activity recognition).

## License
This project is licensed under the [MIT License](LICENSE).