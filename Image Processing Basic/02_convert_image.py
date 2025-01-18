import cv2

#read image
image = cv2.imread('photo2.jpg')

#Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply GaussianBlur
blurred_image = cv2.GaussianBlur(gray_image, (7,5), 0)

#Apply Canny Edge Detection
canny_image = cv2.Canny(blurred_image, 10, 30)

#Apply Threshold
ret, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

#Apply Adaptive Threshold
threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)



# Display the images
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Canny Image', canny_image)
cv2.imshow('Threshold image', threshold_image)
cv2.imshow('Adaptive Threshold image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()