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
        self.V, self.image_idx = None, None

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

    def closest_index(self, f_image):
        '''
        Returns the closest images by facial features by reference
        '''

        if self.V is None:
            f_h5 = "samples/PGAN_attributes.h5"
            with h5py.File(f_h5, "r") as h5:
                self.V = h5["face_vectors"]["data"][...]
                self.image_idx = h5["face_vectors"]["image_idx"][...]

        y = self(f_image)
        dist = np.linalg.norm(self.V - y, ord=1, axis=1)
        idx = np.argsort(dist)

        return self.image_idx[idx]

if __name__ == "__main__":

    clf = latent_face_model()
    
    f_demo = '/home/travis/Desktop/Dw1uV7vWsAEuKwv.jpg'
    idx = clf.closest_index(f_demo)


    import pixelhouse as ph

    A = ph.Canvas().load(f_demo)
    A.show()

    for n in idx[:5]:
        print(n)
        f_closest = f"samples/images/{n:06d}.jpg"
        assert os.path.exists(f_closest)
        B = ph.Canvas().load(f_closest)
        B.show()
