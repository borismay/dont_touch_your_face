import cv2
import winsound
from detectors import *

def single_detect():
    image = cv2.imread(r'c:\Python\face_detect\frames\frame_9.png')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detection_method = 'cnn'
    # boxes = face_recognition.face_locations(rgb,	model=detection_method)
    # encodings = face_recognition.face_encodings(rgb, boxes)
    #

    detection_graph, sess = detector_utils.load_inference_graph()
    boxes, scores = detector_utils.detect_objects(rgb, detection_graph, sess)

    im_height, im_width = rgb.shape[:2]

    detector_utils.draw_box_on_image(2, 0.9,
                                     scores, boxes, im_width, im_height,
                                     gray)

    # for (top, right, bottom, left) in boxes:
    #     cv2.rectangle(gray, (left, top), (right, bottom), (0, 255, 0), 2)

    # cap.release()
    cv2.imshow("Image", gray)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def con_detect():
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cap = cv2.VideoCapture(0)
    # https://stackoverflow.com/questions/11420748/setting-camera-parameters-in-opencv-python
    cap.set(3, 1280)
    cap.set(4, 1024)

    # detection_graph, sess = detector_utils.load_inference_graph()

    HandsDetector = TSDetector()
    FaceDetector = CVLibDetector()

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        # img_hsv[:, :, 2] = clahe.apply(img_hsv[:, :, 2])
        # rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # boxes, scores = detector_utils.detect_objects(rgb, detection_graph, sess)
        # im_height, im_width = rgb.shape[:2]
        # detector_utils.draw_box_on_image(2, 0.5,
        #                                  scores, boxes, im_width, im_height,
        #                                  rgb)
        #
        # cv2.imshow('frame', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        hands = HandsDetector.detect(rgb)
        face = FaceDetector.detect(rgb)

        img_detected = add_objects_to_image(rgb, hands)
        img_detected = add_objects_to_image(img_detected, face, color=(0, 255, 0))
        #
        cv2.imshow('frame', cv2.cvtColor(img_detected, cv2.COLOR_RGB2BGR))

        if objects_touch(face, hands):
            winsound.PlaySound("audiocheck.net_sin_1000Hz_-3dBFS_0.1s.wav", winsound.SND_NOWAIT)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def test_detectors():
    frame = cv2.imread('test1.png')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detection_graph, sess = detector_utils.load_inference_graph()
    HandsDetector = TSDetector()

    im_height, im_width = rgb.shape[:2]
    boxes, scores = detector_utils.detect_objects(rgb, detection_graph, sess)
    hands = HandsDetector.detect(rgb)
    pass



if __name__ == '__main__':
    con_detect()
    # single_detect()
    # test_detectors()