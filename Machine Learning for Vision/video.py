import cv2

# Open video capture
cap = cv2.VideoCapture('example.mp4')  # or use 0 for camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break
    
    # Define the region of interest (ROI)
    x, y, w, h = 100, 100, 200, 200  # example values
    roi = frame[y:y+h, x:x+w]
    
    # Your processing code here

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
