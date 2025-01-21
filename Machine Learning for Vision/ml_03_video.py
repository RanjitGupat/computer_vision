import cv2

# Read a video
cap = cv2.VideoCapture('example_video.mp4')

# Take the first frame of the video
ret, frame = cap.read()

# Setup the initial tracking window
x, y, w, h = 300, 200, 100, 50  # Example values
track_window = (x, y, w, h)

# Set up the ROI for tracking
#In Python, ROI stands for Region of Interest, 
# which is a way to select a specific part of an image or video. 
# ROIs are used in computer vision tasks, such as object detection, 
# transformations, and virtual pens

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply mean shift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # Draw it on the image
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()