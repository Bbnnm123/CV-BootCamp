import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque

model = YOLO("yolov8n.pt")

cap=cv2.VideoCapture(r"c:\Users\ELY\CV BootCamp\Lecture 2\Peoplewalking.mp4")

id_map = {}
nex_id = 1

cv2.namedWindow('People Tracking with Trails', cv2.WINDOW_NORMAL)

trail = defaultdict(lambda: deque(maxlen=30))  # Store last 30 positions for each ID
appear = defaultdict(int)  # Counts appearances of each ID

while True:

    ret, frame = cap.read()
    res = model.track(frame, classes=[0], persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy()

        for box, oid in zip(boxes, ids):    #oid is original id
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = nex_id
                nex_id += 1

            if oid in id_map:
                sid = id_map[oid]   #sid is simplified id (individual)
                trail[sid].append((cx, cy))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {sid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

        # Draw trails
        for sid, positions in trail.items():
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    cv2.line(annotated_frame, positions[i-1], positions[i], (0, 255, 255), 2)
        
        cv2.resizeWindow('People Tracking with Trails', annotated_frame.shape[1], annotated_frame.shape[0])
        cv2.imshow('People Tracking with Trails', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


           