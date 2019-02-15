from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2

#from imutils import face_utils
#import imutils, dlib, os

class FACE_FINDER:
    def __init__(self):
        pass

    def blur_mask(self, img, pts, mask_blur=25):
        hull = cv2.convexHull(pts).squeeze()
        
        h, w = img.shape[:2]

        mask = np.zeros((h, w)).astype(np.uint8)
        mask = cv2.drawContours(mask, [hull], 0, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.dilate(mask, kernel, iterations = 5)
        
        mask = mask.astype(bool)
        mask = 255*mask.astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (mask_blur,)*2, 0)/255

        dx = 15
        ycoord, xcoord = np.where(mask>0)
        x0, x1 = xcoord.min()-dx, xcoord.max()+dx
        y0, y1 = ycoord.min()-dx, ycoord.max()+dx

        mask = mask[y0:y1, x0:x1]
        img = img[y0:y1, x0:x1]
        h, w = mask.shape[:2]
        
        mask = (255*mask.reshape([h, w, 1])).astype(np.uint8)
        #cv2.imshow('image',mask)
        #cv2.waitKey(0)
        #exit()
        
        return img, mask


clf = FACE_FINDER()

def compute(f0, f1):
    img = cv2.imread(f0)
    f_hull = os.path.join(
        'monstergan/base_keypoints/', os.path.basename(f0)).replace(
            '.jpg', '.npy')
    keypoints = np.load(f_hull)

    if EYE_FLAG == "left":
        left_eye = keypoints[36:42]  # left-clockwise
    elif EYE_FLAG == "right":
        right_eye = keypoints[42:48]  # left-clockwise

    img, mask = clf.blur_mask(img, left_eye)
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask.squeeze()
    cv2.imwrite(f1, rgba)

EYE_FLAG = 'left'
    
PIPE = pipeline.Pipeline(
    load_dest = 'monstergan/base_images/',
    save_dest = 'monstergan/left_eye/',
    old_extension = 'jpg',
    new_extension = 'png',
    shuffle=False,
)(compute, -1)


EYE_FLAG = 'right'
    
PIPE = pipeline.Pipeline(
    load_dest = 'monstergan/base_images/',
    save_dest = 'monstergan/right_eye/',
    old_extension = 'jpg',
    new_extension = 'png',
    shuffle=False,
)(compute, -1)
