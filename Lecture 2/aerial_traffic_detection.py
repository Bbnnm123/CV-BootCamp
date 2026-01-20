import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture(r"c:\Users\ELY\CV BootCamp\Lecture 2\AerialTraffic.mp4")

model = YOLO("yolov8l.pt")  # Use larger model for better small object detection

cv2.namedWindow('Aerial Traffic Detection', cv2.WINDOW_NORMAL)

car_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # More aggressive preprocessing for aerial view
    # Resize to larger size for better small object detection
    frame = cv2.resize(frame, (1280, 720))
    
    # Apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Sharpen image to enhance edges
    kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)

    # Run detection with very low confidence
    results = model(frame, classes=[2], conf=0.15, verbose=False)  # Very low confidence for aerial detection

    annotated_frame = results[0].plot()
    
    # Count cars detected
    if results[0].boxes is not None:
        car_count = len(results[0].boxes)
    else:
        car_count = 0
    
    # Add car count to display
    cv2.putText(annotated_frame, f'Cars Detected: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.resizeWindow('Aerial Traffic Detection', annotated_frame.shape[1], annotated_frame.shape[0])
    cv2.imshow('Aerial Traffic Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
