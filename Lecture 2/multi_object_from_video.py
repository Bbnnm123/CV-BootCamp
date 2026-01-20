import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(r"c:\Users\ELY\CV BootCamp\Lecture 2\StreetTraffic.mp4") #been having problems with finding the file, so providing a direct path is the best way

model = YOLO("yolov8n.pt")

cv2.namedWindow('Annotated Video', cv2.WINDOW_NORMAL) #Resizeable window

while True:
    ret, frame = cap.read()

    results = model(frame, classes=[0])  #can filter by class of object, 0 only detects people, can remove the classes to detect everything

    annotated_frame = results[0].plot()
    
    cv2.resizeWindow('Annotated Video', annotated_frame.shape[1], annotated_frame.shape[0]) #Resize window to fit video frame size
    cv2.imshow('Annotated Video', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
