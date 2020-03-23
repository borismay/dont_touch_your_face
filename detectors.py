import os
import cv2
import numpy as np
import face_recognition
# https://www.cvlib.net/
import cvlib as cv
from utils import detector_utils as detector_utils
# pip3 install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow_cpu-2.1.0-cp37-cp37m-win_amd64.whl
import tensorflow as tf
from shapely.geometry import Polygon

############################################################################
class Detector:
    detector_params = {}
    detector = None

    def __init__(self):
        pass

    def set_detector_params(self, params):
        self.detector_params = params

    def detect(self):
        pass


############################################################################
class CVHaarcascadeDetector(Detector):
    def __init__(self, xml_filename):
        self.detector = cv2.CascadeClassifier(xml_filename)

    def detect(self, rgb_img):
        # https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
        # returns an array of bounding boxes (x, y, w, h)
        scale = self.detector_params.get('scale', 1.05)
        min_neighbors = self.detector_params.get('min_neighbors', 6)
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        return self.detector.detectMultiScale(gray_img, scale, min_neighbors)


class FaceDetector(Detector):
    def __init__(self):
        self.detector = face_recognition

    def detect(self, rgb_image):
        detection_method = self.detector_params.get('detection_method', 'cnn')
        # reterns an array of (top, right, bottom, left)
        objects = self.detector.face_locations(rgb_image, model=detection_method)
        # change to an array of (x, y, w, h)
        return [(top, left, bottom - top, right - left) for (top, right, bottom, left) in objects]

class CVLibDetector(Detector):
    def __init__(self):
        self.detector = cv

    def detect(self, rgb_image):
        # returns an array of (top, right, bottom, left)
        objects, confidences = self.detector.detect_face(rgb_image)
        # change to an array of (x, y, w, h)
        return [(top, left, bottom - top, right - left) for (top, right, bottom, left) in objects]


class TSDetector(Detector):
    def __init__(self):
        self.detection_graph, self.sess = detector_utils.load_inference_graph()

    def detect(self, rgb_image):
        # returns (top [0], left [1], bottom [2], right [3])
        boxes, confidences = detector_utils.detect_objects(rgb_image, self.detection_graph, self.sess)

        im_height, im_width = rgb_image.shape[:2]

        detection_th = self.detector_params.get('detection_th', 0.5)
        objects = [(box[0] * im_height, box[3] * im_width, box[2] * im_height, box[1] * im_width) for box, score in zip(boxes, confidences) if score >= detection_th]
        # change to an array of (x, y, w, h)
        return [(int(left), int(top), int(right - left), int(bottom - top)) for (top, right, bottom, left) in objects]


############################################################################
def add_objects_to_image(img_, objects, color=(255, 0, 0)):
    img = np.copy(img_)
    for (x, y, w, h) in objects:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img

def obj_to_poly(obj):
    x, y, w, h = obj
    return Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

def objects_touch(face, hands):
    if face and hands:
        face_poly = obj_to_poly(face[0])
        for hand in hands:
            hand_poly = obj_to_poly(hand)
            if face_poly.intersects(hand_poly):
                return True
    return False

