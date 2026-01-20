import cv2
from ultralytics import YOLO 

model = YOLO("yolov8n.pt")  #pt stands for Pre Trained

image = cv2.imread(r"c:\Users\ELY\CV BootCamp\Lecture 2\mower.jpg")

results = model(image)

annotated_image = results[0].plot()

cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Annotated Image', 3522, 4696)
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

