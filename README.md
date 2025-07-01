# 🛟 Deep Learning-Based Stranded Human Detection in Flood Rescue Operations

This project implements a deep learning-based system to automatically detect stranded humans in flood-affected areas using aerial or drone imagery. Using a custom-trained **YOLOv8-nano** object detection model, the system classifies detected individuals as either **SAFE** or **STRANDED** based on their surrounding context (boats, land patches, deep water, life jackets, or holding objects), and extracts their **geospatial coordinates** for alerting rescue teams.

## 📌 Project Overview

Floods often leave victims stranded in inaccessible regions. Our system aims to:
- **Detect humans and contextual objects** (boat, land, life jacket, etc.) using YOLOv8.
- **Classify each human** as SAFE or STRANDED using bounding box overlap logic.
- **Extract GPS coordinates** from GeoTIFF metadata of satellite/aerial images.
- **Send SMS alerts** with coordinates of stranded individuals to emergency responders.

## 💡 Key Features

- ✅ YOLOv8-nano model for fast and lightweight object detection
- ✅ Custom training on 6 classes: `Human`, `Boat`, `Life_Jacket`, `On_Land`, `Deep_Water`, `Holding_Objects`
- ✅ GeoTIFF coordinate extraction using rasterio
- ✅ Safe/Stranded human classification based on spatial relationships
- ✅ Real-time SMS notification system using Twilio (or similar)


## 🧠 Model Training

The YOLOv8-nano model was trained using **Ultralytics YOLOv8** on a manually labeled dataset using [LabelImg](https://github.com/tzutalin/labelImg).  
We experimented with `nano`, `medium`, and `large` variants. Due to better generalization and reduced overfitting, **YOLOv8-nano** was chosen for deployment.

### 🔗 Download the manually labeled dataset:
[Download datset](https://drive.google.com/drive/folders/1Qpmc1wDor1FfRrml8hNWfoPhYp1TXLGv?usp=drive_link)

### 🔗 Download the fine-tuned YOLOv8-nano model:  
[Download best.pt](https://drive.google.com/file/d/1xQdj77RMevFymHHXfM68vYjmyKCa6Fan/view?usp=drive_link)

### 🔧 Training Details:
- Model: YOLOv8-nano
- Epochs: 80  
- Image size: 640×640  
- Framework: PyTorch, Ultralytics  
- Hardware: Tesla T4 GPU


## 🗂️ File Structure
```bash
project/
│
├── models/
│   └── best.pt          ← Fine-tuned YOLOv8 model
│
├──sample images/
│   └── img1.jgp         ← images with embedded geoTIFFs
│
├── dataset/
│   └── data.yaml        ← Dataset configuration
│   └── train
│   └── test
│   └── valid
│
├── scripts/
│   └── fine_tune.py         ← Training code
│   └── overall_script.py       ← Inference script
│
├── README.md
└── requirements.txt
```

## 🚀 Run the Program
load the fine-tuned model and set the image path in the ```overall_script.py``` file correctly , then run the following command to get the output: 

```bash
pip install -r requirements.txt
py overall_script.py
```


## 📍 Coordinate Extraction (GeoTIFF)

Each .tif image contains geospatial metadata. Using rasterio, pixel coordinates of detected bounding boxes are mapped to real-world latitude/longitude, enabling GPS tracking of victims.


## 🔔 SMS Alert Integration

We used the Twilio API to send real-time alerts to rescue teams. When a STRANDED human is detected, the system sends a message like:

``` 🚨 Stranded person detected at: 17.3850°N, 78.4867°E``` 


## ▶️ Sample Output

**YOLO detection Results:**  
Objects detection for classes `Human`, `Boat`, `Life_Jacket`, `On_Land`, `Deep_Water`, `Holding_Objects`

![image](https://github.com/user-attachments/assets/a5f29ab9-81c0-4e09-a1cd-397761874a6c)


**Classification Results:**  
Humans detected and labeled as `SAFE` (green) or `STRANDED` (red) based on contextual objects.

![image](https://github.com/user-attachments/assets/f4df4a33-32d7-41a4-bdaf-e51a4942c8fa)


**Geo Coordinates:**  
Extracted from the `.tif` file and mapped to latitude/longitude for rescue deployment.

![image](https://github.com/user-attachments/assets/fb9c9e8c-7b41-425f-8cc6-602398ed5668)


## 📚 References

- Ultralytics YOLOv8
- LabelImg - Dataset Annotation Tool
- Rasterio - Geospatial Image Processing
- GeeksforGeeks - Image Processing Concepts
- Roboflow - Dataset Management
- Twilio - SMS API

## 📌 Under the Guidance of 

- Dr. O. B. V. Ramanaiah
- Senior Professor
- JNTUH 


