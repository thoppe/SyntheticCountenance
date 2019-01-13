import numpy as np
import cv2, imutils, dlib, os
from imutils import face_utils
from src.pipeline import Pipeline

model_dest = 'model/dlib'

shape68_pred = dlib.shape_predictor(
    f'{model_dest}/shape_predictor_68_face_landmarks.dat')

def compute_keypoints(img, bbox):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape68 = shape68_pred(gray, bbox)
    keypoints = np.array(face_utils.shape_to_np(shape68))
    
    return keypoints

def compute(f_image, f_fvec):

    img = cv2.imread(f_image)
    f_bbox = f_fvec.replace('/keypoints/','/bbox/')

    if not os.path.exists(f_bbox):
        print(f"MISSING bbox: {f_bbox}")
        return

    coords = np.load(f_bbox)
    bbox = dlib.rectangle(*coords)

    keypoints = compute_keypoints(img, bbox)

    np.save(f_fvec, keypoints)

if __name__ == "__main__":

    P = Pipeline(
        load_dest = 'samples/images',
        save_dest = 'samples/keypoints',
        new_extension = 'npy',
    )(compute, 1)

