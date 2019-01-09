import numpy as np
import cv2, imutils, dlib, os
from imutils import face_utils
from src.pipeline import Pipeline

model_dest = 'model/dlib'

shape5_pred = dlib.shape_predictor(
    f'{model_dest}/shape_predictor_5_face_landmarks.dat')

facerec = dlib.face_recognition_model_v1(
    f'{model_dest}/dlib_face_recognition_resnet_model_v1.dat')

def compute(f_image, f_fvec):

    img = cv2.imread(f_image)

    f_bbox = f_fvec.replace('/face_vectors/','/bbox/')

    if not os.path.exists(f_bbox):
        print(f"MISSING bbox: {f_bbox}")
        return

    bbox = np.load(f_bbox)
    rect = dlib.rectangle(*bbox)

    shape5 = shape5_pred(img, rect)
    fv = facerec.compute_face_descriptor(img, shape5)
    fv = np.array(fv)

    np.save(f_fvec, fv)


P = Pipeline(
    load_dest = 'samples/images',
    save_dest = 'samples/face_vectors',
    new_extension = 'npy',
)(compute, 1)

