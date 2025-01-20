# Image Thresholding is an intensity transformation function in 
# which the values of pixels below a particular threshold are reduced, 
# and the values above that threshold are boosted.  
# This generally results in a bilevel image at the end, 
# where the image is composed of black and white pixels. 
# Thresholding belongs to the family of point-processing techniques. 
# In this article, you will learn how to perform Image Thresholding in OpenCV.

import cv2
import numpy as np

# Read the image
image = cv2.imread('photo2.jpg')

# Check if the image is loaded
if image is None:
    print('Could not open or find the image', 'photo2.jpg')
    exit(0)

# Convert the image to grayscale
gray_image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply simple thresholding
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Display the thresholded image
cv2.imshow('Thresholded Image', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()