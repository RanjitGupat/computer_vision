# SIFT (Scale Invariant Feature Transform) 
# Detector is used in the detection of interest points on an input image. 
# It allows the identification of localized features in images 
# which is essential in applications such as: 
# Object Recognition in Images
# Path detection and obstacle avoidance algorithms
# Gesture recognition, Mosaic generation, etc.


import cv2
import numpy as np 

#load the image 
path = 'photo2.jpg'
image = cv2.imread(path)

#Check if the image is loaded
if image is None:
    print('Could not open or find the image', path)
    exit(0)

#Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Initialize the SIFT detector
sift = cv2.SIFT_create()

#Detect the keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

#Draw the keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with SIFT Keypoints', image_with_keypoints)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()