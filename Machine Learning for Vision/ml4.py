import cv2

# Load left and right images
imgL = cv2.imread('photo2.jpg', 0)
imgR = cv2.imread('photo2.jpg', 0)

# Create a stereo block matcher object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Compute the disparity map
disparity = stereo.compute(imgL, imgR)

# Display the disparity map
cv2.imshow('Disparity', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()