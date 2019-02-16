from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2
from bridson import poisson_disc_samples

save_dest = 'monstergan/eye_movie'
os.system(f'mkdir -p {save_dest}')

def combine_images(img0, img1, alpha):
    dst = (1-alpha)*img1 + alpha*img0
    return np.clip(dst, 0, 255).astype(np.uint8)

#width, height = 1920, 1080
height, width = 1080//4, 1920//4
radius = 30
random_seed = 42


pts = poisson_disc_samples(width=width, height=height, r=radius)
pts = np.array(pts).astype(int)

'''
#pts = pts[:, [1,0]]
canvas = np.zeros((int(height), int(width), 3))
for (x,y) in pts:
    cv2.circle(canvas, (x,y), radius=10, color=[255,255,255])

cv2.imshow('image', canvas)
cv2.waitKey(0)
'''



for frame_idx in range(90):
    F_IMG0 = sorted(glob.glob(f"monstergan/left_eye/????_{frame_idx:06d}.png"))
    F_IMG1 = sorted(glob.glob(f"monstergan/right_eye/????_{frame_idx:06d}.png"))
    F_IMG = F_IMG0 + F_IMG1

    np.random.seed(82)
    np.random.shuffle(F_IMG)

    


    n_eyes = min([len(pts), len(F_IMG)])
    F_IMG = F_IMG[:n_eyes]
    pts = pts[:n_eyes]
    print(f"Using {len(pts)} eyes")


    canvas = np.zeros((height, width, 3))

    for f_img, (x,y) in zip(F_IMG, pts):

        #cv2.circle(canvas, (x,y), radius=10, color=[255,255,255])
        #continue

        img = cv2.imread(f_img, cv2.IMREAD_UNCHANGED)
        img, alpha = img[:, :, :3], img[:, :, 3]
        alpha = alpha.astype(float)/255

        mean_pixel = img.mean(axis=0).mean(axis=0).astype(np.uint8)
        #mean_pixel = [0,0,0]

        imgx = np.ones((height,width,3))*mean_pixel
        soft_mask = np.zeros((height,width, 1))

        h, w = img.shape[:2]
        imgx[:h, :w] = img
        soft_mask[:h, :w] = alpha.reshape((h,w,1))

        x += w//2
        y += h//2

        imgx = np.roll(imgx, x, axis=1)
        imgx = np.roll(imgx, y, axis=0)
        soft_mask = np.roll(soft_mask, x, axis=1)
        soft_mask = np.roll(soft_mask, y, axis=0)

        canvas = combine_images(imgx, canvas, soft_mask)

    f_save = os.path.join(save_dest, f"{frame_idx:06d}.jpg")
    print(f_save)
    cv2.imwrite(f_save, canvas)

    #cv2.imshow('image', canvas)
    #cv2.waitKey(0)

