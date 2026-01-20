
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  #v8 nano best for fidgeting

#webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open Webcam")
    exit()

print("Webcam opened")
print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam Closed")