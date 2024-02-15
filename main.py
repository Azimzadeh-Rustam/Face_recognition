import cv2

path_cascade = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path_cascade)
color = (255, 0, 0)

capture = cv2.VideoCapture(0)
capture.set(3, 500)
capture.set(4, 300)

while True:
    success, img = capture.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(20,20))

    for (x, y, width, height) in faces:
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness=2)

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
