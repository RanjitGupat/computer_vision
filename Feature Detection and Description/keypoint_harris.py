
# Harris Corner detection algorithm was developed to identify the internal corners of an image. 
# The corners of an image are basically identified as the regions in 
# which there are variations in large intensity of the gradient in all possible dimensions and directions. 
# Corners extracted can be a part of the image features, 
# which can be matched with features of other images, and can be used to extract accurate information. 
# Harris Corner Detection is a method to extract the corners from the input image 
# and to extract features from the input image. 


import cv2
import numpy as np

#Read an image
image = cv2.imread('photo2.jpg')

#Check if the image is loaded
if image is None:
    print('Could not open or find the image', 'photo2.jpg')
    exit(0)

#Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Perform Harris corner detection
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)

#Dilate the corner points to make them more visible
dst = cv2.dilate(dst, None)

#Threshold for an optimal value, it may vary depending on the image
image[dst > 0.01 * dst.max()] = [0, 0, 255] # Mark detected corners in red

# Display the image with marked corners
cv2.imshow('Harris Corner Detection', image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()