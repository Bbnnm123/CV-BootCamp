import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(r"c:\Users\ELY\CV BootCamp\Lecture 2\StreetTraffic.mp4")

unique_IDs = set()

cv2.namedWindow('Object Tracking', cv2.WINDOW_NORMAL) #resizeable window

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, classes=[2], persist=True, verbose=False, conf=0.5, iou=0.7, tracker="botsort.yaml")  #class 2 is for cars, verbose false disables the constant printing in the terminal
    annotated_frame = results[0].plot()                     #conf=0.5, iou=0.7, tracker="botsort.yaml are parameters for precise tracking performance, confidence and IOU for stricter matching, Intersection over Union (IOU) is the Area of Intersection/Area of Union ~~ 1. If IOU > 0.5, it is considered a match. Harder when objects are fast moving or close.

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_IDs.add(oid)
    
    cv2.putText(annotated_frame, f'Count: {len(unique_IDs)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    #(10,30) is the location of the text
    cv2.resizeWindow('Object Tracking', annotated_frame.shape[1], annotated_frame.shape[0]) #Resize window
    cv2.imshow('Object Tracking', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()


