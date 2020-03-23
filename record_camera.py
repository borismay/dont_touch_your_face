import numpy as np
import cv2
import os
import time

cap = cv2.VideoCapture(0)
FRAME_PATH = os.path.join(os.getcwd(), 'frames')

frame_num = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imwrite(os.path.join(FRAME_PATH, 'frame_{}.png'.format(frame_num)), frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1
    time.sleep(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()