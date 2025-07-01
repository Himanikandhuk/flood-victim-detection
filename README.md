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
├── data/
│   └── data.yaml        ← Dataset configuration
│
├── scripts/
│   └── train.py         ← Training code
│   └── predict.py       ← Inference script
│
├── README.md
└── requirements.txt
```


## ▶️ Sample Output

**Classification Results:**  
Humans detected and labeled as `SAFE` (green) or `STRANDED` (red) based on contextual objects.

**Geo Coordinates:**  
Extracted from the `.tif` file and mapped to latitude/longitude for rescue deployment.


## 🚀 Run the Program

```bash
pip install -r requirements.txt
py detect_and_classify.py
```


## 📍 Coordinate Extraction (GeoTIFF)

Each .tif image contains geospatial metadata. Using rasterio, pixel coordinates of detected bounding boxes are mapped to real-world latitude/longitude, enabling GPS tracking of victims.


## 🔔 SMS Alert Integration

We used the Twilio API to send real-time alerts to rescue teams. When a STRANDED human is detected, the system sends a message like:

``` 🚨 Stranded person detected at: 17.3850°N, 78.4867°E``` 


## 📚 References

Ultralytics YOLOv8
LabelImg - Dataset Annotation Tool
Rasterio - Geospatial Image Processing
GeeksforGeeks - Image Processing Concepts
Roboflow - Dataset Management
Twilio - SMS API

## 🙋‍♀️ Contributors

Supraja Sandhya Thota
Himani Kandhukoori 
Afrah Anaan
Ayesha Fatima

Guided by: Dr. O. B. V. Ramanaiah
           Senior Professor 
           JNTUH 


