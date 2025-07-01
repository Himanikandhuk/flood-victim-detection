from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np

# Load model
model = YOLO("/content/drive/My Drive/floodsnew/best.pt")

# Load GeoTIFF
geotiff_path = '/content/drive/My Drive/floodsnew/output_geotiff_kerala.tif'
with rasterio.open(geotiff_path) as dataset:
    transform = dataset.transform
    crs = dataset.crs
    # Read RGB bands and reshape as image
    image_array = dataset.read([1, 2, 3])  # Assuming RGB bands are 1,2,3
    image_array = reshape_as_image(image_array)

# Convert image to BGR (YOLO and OpenCV use BGR)
image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

# Save temporary JPG for YOLO (optional) or use array directly
temp_image_path = '/content/temp_image.jpg'
cv2.imwrite(temp_image_path, image_bgr)

# Run YOLO prediction
results = model.predict(temp_image_path, conf=0.25)

class_names = ['Humans', 'Boat', 'Life_Jacket', 'On_Land', 'Deep_Water', 'Holding_Objects']
human_class_id = class_names.index('Humans')

def classify_human(human_box, all_boxes):
    safe = 0
    stranded = 0
    x1, y1, x2, y2 = human_box['xyxy']

    for box in all_boxes:
        if box['cls'] == human_class_id:
            continue
        bx1, by1, bx2, by2 = box['xyxy']
        overlap = not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2)
        if overlap:
            label = box['label']
            if label in {'Boat', 'Life_Jacket', 'On_Land'}:
                safe += 1
            elif label in {'Deep_Water', 'Holding_Objects'}:
                stranded += 1
    return 'SAFE' if safe > stranded else 'STRANDED'

# Extract detections
all_boxes = []
for box in results[0].boxes:
    cls = int(box.cls[0])
    label = class_names[cls]
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    all_boxes.append({'cls': cls, 'label': label, 'xyxy': xyxy})

# Process results
stranded_coords = []
for box in all_boxes:
    if box['label'] == 'Humans':
        classification = classify_human(box, all_boxes)
        x1, y1, x2, y2 = box['xyxy']

        if classification == 'STRANDED':
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            lon, lat = rasterio.transform.xy(transform, center_y, center_x)
            stranded_coords.append((lat, lon))

# Print stranded human coordinates
print("🚨 Stranded human locations:")
for lat, lon in stranded_coords:
    print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")

# Draw bounding boxes and labels
for box in all_boxes:
    if box['label'] == 'Humans':
        classification = classify_human(box, all_boxes)
        color = (0, 255, 0) if classification == 'SAFE' else (0, 0, 255)
        x1, y1, x2, y2 = box['xyxy']
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, classification, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Save and show
output_path = '/content/classified_output.jpg'
cv2.imwrite(output_path, image_bgr)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Human Classification: SAFE / STRANDED")
plt.show()
