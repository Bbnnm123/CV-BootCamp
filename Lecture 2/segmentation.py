import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(r"c:\Users\ELY\CV BootCamp\Lecture 2\Peoplewalkingbusy.mp4")

cv2.namedWindow('Segmented People', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    results = model.track(source=frame, classes=[0], persist=True, verbose=False) 
    
    for r in results:
        annotated_frame = frame.copy()
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.numpy()
            boxes = r.boxes.xyxy.numpy()
            ids = r.boxes.id.numpy()

            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1, y1, x2, y2 = map(int, boxes[i])
                mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (frame.shape[1], frame.shape[0]))
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, f'ID: {person_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.resizeWindow('Segmented People', annotated_frame.shape[1], annotated_frame.shape[0])
            cv2.imshow('Segmented People', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
cap.release()
cv2.destroyAllWindows()

#figure out whats wrong with shutting down windows properly with q, and also how to make the contours more accurate.