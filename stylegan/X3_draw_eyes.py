from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2


frame_idx = 0
F_IMG = glob.glob("monstergan/left_eye/????_*.png")
print(F_IMG)

