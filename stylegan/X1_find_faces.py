from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2

from imutils import face_utils
import imutils, dlib, os

class FACE_FINDER:
    def __init__(self):

        model_dest = "model/dlib"
        self.shape68_pred = dlib.shape_predictor(
            f"{model_dest}/shape_predictor_68_face_landmarks.dat"
        )
        self.detector = dlib.get_frontal_face_detector()
        
    def compute_keypoints(self, img, bbox):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape68 = self.shape68_pred(gray, bbox)
        keypoints = np.array(face_utils.shape_to_np(shape68))

        return keypoints

    def compute_bbox(self, img, n_upsample=0):
        faces = self.detector(img, n_upsample)

        if len(faces) == 0:
            logger.warning(f"No faces found in image! Upsampling")
            faces = self.detector(img, 2)
            
            if len(faces) == 0:
                logger.error(f"Still can't find a face")
                raise ValueError
                        
        if len(faces) != 1:
            
            face_size = [face.area() for face in faces]
            logger.warning(
                f"Found {len(faces)} faces in image! Sizes {face_size}")
            idx = np.argmax(np.array(face_size))
            return faces[idx]

        return faces[0]

    def keypoints(self, img):
        bbox = self.compute_bbox(img)
        pts = self.compute_keypoints(img, bbox)
        return pts

    def convex_hull(self, img):
        bbox = self.compute_bbox(img)
        pts = self.compute_keypoints(img, bbox)
        hull = cv2.convexHull(pts).squeeze()

        #kernel = np.ones((5,5),np.uint8)
        #expand = cv2.dilate(hull, kernel, iterations = 5)
        #return expand
    
        return hull

    def blur_mask(self, img, mask_blur=201):
        h, w = img.shape[:2]
        
        hull = self.convex_hull(img)
        mask = np.zeros((h, w)).astype(np.uint8)
        mask = cv2.drawContours(mask, [hull], 0, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.erode(mask, kernel, iterations = 20)

        mask = mask.astype(bool)
        mask = 255*mask.astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (mask_blur,)*2, 0)/255
        mask = mask.reshape([h, w, 1])

        #cv2.imshow('image',mask)
        #cv2.waitKey(0)
        #exit()
        
        return mask


clf = FACE_FINDER()

def compute(f0, f1):
    img = cv2.imread(f0)
    try:
        pts = clf.keypoints(img)
    except Exception as EX:
        logger.error("Completely failed with {f0} {EX}")
        return False

    np.save(f1, pts)

PIPE = pipeline.Pipeline(
    load_dest = 'monstergan/base_images/',
    save_dest = 'monstergan/base_keypoints/',
    new_extension = 'npy',
    old_extension = 'jpg',
    shuffle=False,
)(compute, 1)
