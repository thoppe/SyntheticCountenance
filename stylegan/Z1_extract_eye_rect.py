from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2

def compute_left(f0, f1):
    img = cv2.imread(f0)

    delta_y = 80
    delta_x = int(delta_y*(16/9.0))
    start_y = 445

    #if EYE_FLAG == "left":
    start_x = 305
    #elif EYE_FLAG == "right":
    #    start_x = 570
    
    y0, y1 = [start_y, start_y+delta_y]
    x0, x1 = [start_x, start_x+delta_x]
    img = img[y0:y1, x0:x1]
    cv2.imwrite(f1, img)

def compute_right(f0, f1):
    img = cv2.imread(f0)

    delta_y = 80
    delta_x = int(delta_y*(16/9.0))
    start_y = 445

    start_x = 570
    
    y0, y1 = [start_y, start_y+delta_y]
    x0, x1 = [start_x, start_x+delta_x]
    img = img[y0:y1, x0:x1]
    cv2.imwrite(f1, img)
    

EYE_FLAG = 'left'
    
PIPE = pipeline.Pipeline(
    load_dest = 'tripgan/base_images/',
    save_dest = 'tripgan/left_eye/',
    old_extension = 'jpg',
    new_extension = 'jpg',
    shuffle=True,
)(compute_left, -1)


EYE_FLAG = 'right'
    
PIPE = pipeline.Pipeline(
    load_dest = 'tripgan/base_images/',
    save_dest = 'tripgan/right_eye/',
    old_extension = 'jpg',
    new_extension = 'jpg',
    shuffle=True,
)(compute_right, -1)
