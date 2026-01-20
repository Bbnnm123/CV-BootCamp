import cv2

img = cv2.imread('image_name.jpg') #change to actual image name

resizing = cv2.resize(img, (640, 480))  
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurring = cv2.GaussianBlur(img, (2, 2), 0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Resized Image', resizing)
cv2.imshow('Gray Image', grey)
cv2.imshow('Blurred Image', blurring)
cv2.imshow('Edge Detected Image', edges) 
cv2.waitKey(0)  
cv2.destroyAllWindows()

