import cv2

# image file
img_file = 'Source\\cars.jpg'

# pretrained car classifier
tracker = cv2.CascadeClassifier('trackingAlgo.xml')

# open image in openCV
img = cv2.imread(img_file)

# convert to grayscale image
grayScale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
car_tracker = tracker.detectMultiScale(grayScale_img)

# draw ractangle
for (x, y, w, h) in car_tracker:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow("Car Detactor", img)
cv2.waitKey()
