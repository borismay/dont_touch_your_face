import numpy as np
import cv2
import os

cap = cv2.VideoCapture(0)
FRAME_PATH = os.path.join(os.getcwd(), 'frames')
face_cascade = cv2.CascadeClassifier('c:\Python\env381\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('c:\Python\env381\Lib\site-packages\cv2\data\haarcascade_eye.xml')


def detect_face(frame_img, gray_img):
    faces = face_cascade.detectMultiScale(gray_img, 1.05, 5)
    for (x, y, w, h) in faces:
        frame_img = cv2.rectangle(frame_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame_img

        # roi_gray = gray[y:y + h, x:x + w]
        # roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in eyes:
        #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

frame_num = 0
# while(frame_num < 100):
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imwrite(os.path.join(FRAME_PATH, 'frame_{}.png'.format(frame_num)), gray)
    frame_face = detect_face(frame, gray)
    # cv2.imwrite(os.path.join(FRAME_PATH, 'frame_face_{}.png'.format(frame_num)), frame_face)

    cv2.imshow('frame', frame_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()