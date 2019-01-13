import numpy as np
import os
import cv2, imutils, dlib
from imutils import face_utils
from src.pipeline import Pipeline

print(f"dlib CUDA status: {dlib.DLIB_USE_CUDA}")
model_dest = 'model/dlib'
detector = dlib.get_frontal_face_detector()

def compute_bbox(img, n_upsample=0):
    return detector(img, n_upsample)

def compute(f_image, f_bbox, n_upsample=0):

    img = cv2.imread(f_image)
    faces = compute_bbox(img, n_upsample)

    if len(faces) != 1:
        print(f"REMOVING: {f_image}, {len(faces)} faces detected.")
        return os.remove(f_image)

    face = faces[0]
    bbox = [face.left(), face.top(), face.right(), face.bottom()]
    bbox = np.array(bbox)

    #print(f"Computed bbox {f_bbox}")
    np.save(f_bbox, bbox)

if __name__ == "__main__":
    P = Pipeline(
        load_dest = 'samples/images',
        save_dest = 'samples/bbox',
        new_extension = 'npy',
    )(compute, -1)

