import cv2

#Load the data pre-trained model haar casecade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Read the image
image = cv2.imread('photo2.jpg')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces
face = face_cascade.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
 
#Draw rectangel around the faces
for(x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x+h, y+w), (255, 0, 0), 2) 

#Display the image with the detected faces
cv2.imshow('detected_faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
