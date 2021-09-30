import cv2
# load some some pre-trained data front face from openCV Haar cascade algorithm
car_tracker = cv2.CascadeClassifier('trackingAlgo.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# webcam = cv2.VideoCapture("newYork.mp4")


# capture video from webcam
webcam = cv2.VideoCapture('Source\\pedestrians-car.mp4')

while True:
    # reade current frame
    successful_frame_read, frame = webcam.read()
    if successful_frame_read:
        grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    # DETECT FACES
    Car_cordinations = car_tracker.detectMultiScale(grayscale_img)
    pedestrian_cordinations = pedestrian_tracker.detectMultiScale(
        grayscale_img)

    # draw rectangle around the face and multiple faces
    for (x, y, w, h) in Car_cordinations:
        cv2.rectangle(frame, (x + 1, y + 1), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrian_cordinations:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # press Q key for quit
    if key == 81 or key == 113:
        break

# release the webcam
webcam.release()
