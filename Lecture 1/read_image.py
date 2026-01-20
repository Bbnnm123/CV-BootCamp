import cv2

img1 = cv.imread('image_name.jpg') #change to actual image name
cv2.imshow('The original image', img1)
cv2.waitKey(0)

cv2.imwrite('saved_original_image.jpg', img1) #change to desired output image name

