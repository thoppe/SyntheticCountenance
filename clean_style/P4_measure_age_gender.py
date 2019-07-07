"""
https://github.com/yu4u/age-gender-estimation

Keras implementation of a CNN network for age and gender estimation

Download the model here and place it in model/age_gender/
https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5
"""

from src import pipeline
from src.logger import logger
import numpy as np
import json
import cv2
import os

import pixelhouse as ph
from tqdm import tqdm
import dlib
import tensorflow as tf
from model.age_gender.wide_resnet import WideResNet


use_GPU = False
device = "/GPU:0" if use_GPU else "/CPU:0"

# Load model and weights
with tf.device(device):
    img_size, depth, width = 64, 16, 8
    model = WideResNet(img_size, depth=depth, k=width)()
    f_weights = "model/age_gender/weights.28-3.73.hdf5"
    model.load_weights(f_weights)

    detector = dlib.get_frontal_face_detector()



def compute(f_img, f1):

    if os.path.exists(f1):
        return False

    img = ph.load(f_img).rgb
  
    target_size = (64, 64)
    detected = detector(img, 1)

    item = {"f_img": f_img}
    item['n_faces'] = len(detected)
    #item['bbox_faces'] = detected

    img_w, img_h = img.shape[0], img.shape[1]

    margin = 0.4
    
    if len(detected) == 1:
        d = detected[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
        w, h = d.width(), d.height()
        
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), img_w - 1)
        yw2 = min(int(y2 + margin * h), img_h - 1)
        face = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], target_size)
        faces = np.expand_dims(face, 0)

        results = model.predict(faces)
        item['female_score'] = float(results[0].ravel()[0])
        ages = np.arange(0, 101).reshape(101, 1)
        item['age'] = float(results[1].dot(ages).flatten().ravel()[0])

    js = json.dumps(item, indent=2)

    with open(f1, "w") as FOUT:
        FOUT.write(js)


if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest="data/images/",
        save_dest="data/AgeGender_score/",
        new_extension="json",
        old_extension="jpg",
        shuffle=True,
    )(compute, 1)
