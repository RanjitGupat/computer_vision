import cv2
import numpy as np

#Load image
img_path = 'photo2.jpg'
image = cv2.imread(img_path)

if image is None:
    print(f'Could not open or find the image: ', img_path)
    exit(0)

#Convert the image to gray scale
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Perform canny edge detection 
edges  = cv2.Canny(grey_image, 100, 200)  #threshold1=100, threshold2=200  Adjust thresholds as needed


#Display the image with the detected edges
cv2.imshow('Original Image', image)
cv2.imshow('Detected Edges', edges)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

