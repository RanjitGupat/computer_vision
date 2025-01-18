import cv2

#read image
image = cv2.imreaed('icon1.png')

#display the image
cv2.imshow('example of image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
