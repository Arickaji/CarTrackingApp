import cv2
# load some some pre-trained data front face from openCV Haar cascade algorithm
trained_data = cv2.CascadeClassifier('trackingAlgo.xml')

# webcam = cv2.VideoCapture("newYork.mp4")


# capture video from webcam
webcam = cv2.VideoCapture('Source\\Full Self-Driving.mp4')

while True:
    # reade current frame
    successful_frame_read, frame = webcam.read()
    if successful_frame_read:
        grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # DETECT FACES
    cordinations = trained_data.detectMultiScale(grayscale_img)

    # draw rectangle around the face and multiple faces
    for (x, y, w, h) in cordinations:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # press Q key for quit
    if key == 81 or key == 113:
        break

# release the webcam
webcam.release()
