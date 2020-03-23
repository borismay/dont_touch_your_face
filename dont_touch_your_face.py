import cv2
import winsound
from detectors import *


def con_detect():
    cap = cv2.VideoCapture(0)
    # https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    cap.set(3, 1280/2)
    cap.set(4, 1024/2)

    HandsDetector = TSDetector()
    FaceDetector = CVLibDetector()

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands = HandsDetector.detect(rgb)
        face = FaceDetector.detect(rgb)

        img_detected = add_objects_to_image(rgb, hands)
        img_detected = add_objects_to_image(img_detected, face, color=(0, 255, 0))

        cv2.imshow('frame', cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR))

        if objects_touch(face, hands):
            winsound.PlaySound("audiocheck.net_sin_1000Hz_-3dBFS_0.1s.wav", winsound.SND_NOWAIT)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    con_detect()
