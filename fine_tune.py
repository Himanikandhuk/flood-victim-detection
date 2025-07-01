from ultralytics import YOLO


model = YOLO("yolov8n.pt")

# Train the model
model.train(data="/content/drive/My Drive/floodsnew/data.yaml",
            epochs=80,  # Reduce to 50 for faster training
            imgsz=640,  # Keep image size
            batch=8,  # Use larger batch size since GPU is available
            workers=2,  # Enable multiple threads for faster loading
            device='cuda')
