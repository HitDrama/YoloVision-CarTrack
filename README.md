# 🚗 YoloVision-CarTrack

**YoloVision-CarTrack** is a computer vision project that uses YOLOv8 and ByteTrack to detect, track, and analyze vehicles in real-time video streams. It can count vehicles entering and leaving a monitored zone and estimate their speed based on frame movement.

---

## 🧠 Key Features

- 🎯 Detects vehicles (cars, motorcycles, buses, trucks) using YOLOv8
- 🧭 Estimates speed (km/h) based on vertical movement and FPS
- 🔄 Tracks individual vehicles using ByteTrack
- 📊 Counts vehicles entering and leaving a detection line
- ⚙️ Tuned detection for motorcycles via scale and confidence adjustments

---

## ⚙️ Technologies Used

- 🧠 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 🎯 OpenCV
- 🔍 Supervision
- 🐍 Python 3.8+

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/YoloVision-CarTrack.git
cd YoloVision-CarTrack

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # macOS/Linux
venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

- Run the system locally using:
```bash
python run.py
```
- Link URL:
```bash
[python run.py](http://127.0.0.1:5000/yolo)
```

---

## 🖼️ Image Processing Workflow
Open video and downscale frames for faster processing.

1. Draw a horizontal line at 70% of frame height to detect direction of movement (entry/exit).

2. Run YOLOv8 with selected vehicle class IDs (2: car, 3: motorcycle, 5: bus, 7: truck).

3. Track each object using ByteTrack for consistent IDs across frames.

4. Estimate speed by computing pixel movement per frame and converting to km/h.

5. Detect crossing direction:

6. From top to bottom → Entering

7. From bottom to top → Leaving

8. Draw overlays: bounding boxes, vehicle IDs, speed, and total counts on the frame.

---

## 🎥 Demo Video

https://github.com/user-attachments/assets/30bc601b-f7c0-4a6f-b026-769fe4ee3f9b

## 👨‍💻 Developer

Dang To Nhan

