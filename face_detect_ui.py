import streamlit as st
import glob
from detectors import *


##############################
if __name__ == '__main__':
    IMAGES_PATH = os.path.join(os.getcwd(), 'frames')
    CLASSIFIERS_PATH = r'c:\Python\env381\Lib\site-packages\cv2\data'

    images_path = st.sidebar.text_input('Images path:', IMAGES_PATH)
    image_files = glob.glob(os.path.join(images_path, '*.png'))

    choices = [os.path.split(filename)[-1] for filename in image_files]

    image_filename = st.sidebar.selectbox('Image filename', choices)

    classifiers = [
        r'haarcascade_frontalface_default.xml',
        r'haarcascade_frontalface_alt_tree.xml',
        r'haarcascade_frontalface_alt2.xml',
        r'haarcascade_frontalface_alt.xml',
        r'haarcascade_eye_tree_eyeglasses.xml',
        r'haarcascade_eye.xml',
        # r'hand.xml',
        r'face_recognition',
        r'cvlib'
    ]

    classifier_name = st.sidebar.selectbox('Classifier', classifiers, index=0)

    frame = cv2.imread(os.path.join(images_path, image_filename), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # cascade_class = cv2.CascadeClassifier()
    # detector_param_scale = st.sidebar.slider('detect_scale', 1.05, 1.6, value=1.05, step=0.05)
    # detector_param_min_neighbors = st.sidebar.slider('detect_min_n', 3, 6, value=5, step=1)

    if classifier_name.startswith('haarcascade'):
        classifier = CVHaarcascadeDetector(os.path.join(CLASSIFIERS_PATH, classifier_name))
        classifier.set_detector_params(
            {
                'scale': st.sidebar.slider('detect_scale', 1.05, 1.6, value=1.05, step=0.05),
                'min_neighbors': st.sidebar.slider('detect_min_n', 3, 6, value=5, step=1)
            }
        )
    elif classifier_name.startswith('face'):
        classifier = FaceDetector()
        classifier.set_detector_params(
            {
                'detection_method': st.sidebar.radio('Detection method', ('hog', 'cnn'))
            }
        )
    elif classifier_name.startswith('cvlib'):
        classifier = CVLibDetector()

    detected = add_objects_to_image(rgb, classifier.detect(rgb))

    st.image(rgb)
    st.image(detected)
