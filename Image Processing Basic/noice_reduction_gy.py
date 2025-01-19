import cv2

#Load the data pre-trained model haar casecade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image
image_path = 'photo2.jpg'
image = cv2.imread(image_path)


#Detect faces
face = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

# Check if image loading was successful
if image is None:
    print(f'Failed to load image at path: {image_path}')
    exit()

#Draw rectangel around the faces
for(x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x+h, y+w), (255, 0, 0), 2) 
# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (15, 5), 0)  # Adjust kernel size (5, 5) as needed

# Display the original image and the blurred image
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()