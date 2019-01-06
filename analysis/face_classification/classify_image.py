import sys, json

import cv2
from keras.models import load_model
import numpy as np

from .utils.datasets import get_labels
from .utils.inference import detect_faces
from .utils.inference import draw_text
from .utils.inference import draw_bounding_box
from .utils.inference import apply_offsets
from .utils.inference import load_detection_model
from .utils.inference import load_image
from .utils.preprocessor import preprocess_input

model_dest = 'face_classification/trained_models'

f_detection = f'{model_dest}/detection_models/haarcascade_frontalface_default.xml'
f_emotion = f'{model_dest}/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
f_gender = f'{model_dest}/gender_models/simple_CNN.81-0.96.hdf5'

emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(f_detection)
emotion_classifier = load_model(f_emotion, compile=False)
gender_classifier = load_model(f_gender, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]


def classify(f_image):

    rgb_image = load_image(f_image, grayscale=False)
    gray_image = load_image(f_image, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    print(rgb_image.shape)

    face_data = []
    
    faces = detect_faces(face_detection, gray_image)
    print(faces)

    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except Exception as EX:
            print(f"Warning: {EX}")
            continue

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        print(face_coordinates, gender_text, emotion_text)

        face_data.append({
            'bounding_box' : face_coordinates.tolist(),
            'gender_vector' : gender_prediction.tolist(),
            'gender_labels' : gender_labels,
            'emotion_vector' : emotion_prediction.tolist(),
            'emotion_labels' : emotion_labels,
        })

    return face_data


if __name__ == "__main__":
    f_image = '000260.jpg'

    res = classify(f_image)
    print(res)
    print(json.dumps(res,indent=2))
