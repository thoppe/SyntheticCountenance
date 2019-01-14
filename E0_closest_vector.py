import numpy as np
import os, json, glob, h5py
import pandas as pd
from tqdm import tqdm
import cv2, imutils, dlib


class latent_face_model:
    def __init__(self):
        model_dest = "model/dlib"

        self.shape5_pred = dlib.shape_predictor(
            f"{model_dest}/shape_predictor_5_face_landmarks.dat"
        )

        self.facerec = dlib.face_recognition_model_v1(
            f"{model_dest}/dlib_face_recognition_resnet_model_v1.dat"
        )

        self.detector = dlib.get_frontal_face_detector()

    def __call__(self, f_image, n_upsample=0):

        assert os.path.exists(f_image)
        img = cv2.imread(f_image)

        faces = self.detector(img, n_upsample)

        if len(faces) != 1:
            print(f"Not a single face in the image!")
            return None

        bbox = faces[0]

        shape5 = self.shape5_pred(img, bbox)
        fv = self.facerec.compute_face_descriptor(img, shape5)
        fv = np.array(fv)

        return fv


f_h5 = "samples/PGAN_attributes.h5"
h5 = h5py.File(f_h5, "r")

V = h5["face_vectors"]["data"][...]
image_idx = h5["face_vectors"]["image_idx"][...]

clf = latent_face_model()

# f_demo = 'src/000260.jpg'
# f_demo = '/home/travis/Desktop/20170702_171733.jpg'
# f_demo = '/home/travis/Desktop/the-10-most-talked-about-celebrities-during-the-grammys.jpg'
# f_demo = '/home/travis/Desktop/girl-glowing-skin-blue-eyes.jpg'
f_demo = "hoppe.jpg"


y = clf(f_demo)

dist = np.linalg.norm(V - y, ord=1, axis=1)
idx = np.argsort(dist)


import pixelhouse as ph

A = ph.Canvas().load(f_demo)
A.show()

for n in image_idx[idx][:5]:
    print(n)
    f_closest = f"samples/images/{n:06d}.jpg"
    assert os.path.exists(f_closest)
    B = ph.Canvas().load(f_closest)
    B.show()

# ph.hstack([A, B]).show()
