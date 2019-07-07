"""
https://github.com/justinshenk/fer

Adapted from
Facial Expression Recognition with a deep neural network as a PyPI package
"""

from src import pipeline
from src.logger import logger
import numpy as np
import json
import cv2
import os

import pixelhouse as ph
from tqdm import tqdm
import tensorflow as tf
import dlib
from keras.models import load_model

#from model.age_gender.wide_resnet import WideResNet
f_model = 'model/fer/emotion_model.hdf5'

use_GPU = False
device = "/GPU:0" if use_GPU else "/CPU:0"
detector = dlib.get_frontal_face_detector()

# Load model and weights
with tf.device(device):
    
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = False
    model = load_model(f_model, compile=False)

    # Should be 64, 64
    target_size = model.input_shape[1:3]
    model._make_predict_function()

def compute(f_img, f1):

    if os.path.exists(f1):
        return False

    img = ph.load(f_img).rgb
    detected = detector(img, 1)

    item = {"f_img": f_img}
    item['n_faces'] = len(detected)
    img_w, img_h = img.shape[0], img.shape[1]

    margin = 0.4

    labels = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'sad',
        5: 'surprise',
        6: 'neutral'
    }
    
    if len(detected) == 1:
        d = detected[0]
        x1, y1, x2, y2 = d.left(), d.top(), d.right() + 1, d.bottom() + 1
        w, h = d.width(), d.height()
        
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), img_w - 1)
        yw2 = min(int(y2 + margin * h), img_h - 1)
        face = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], target_size)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, -1)
        faces = np.expand_dims(face, 0)

        faces = faces.astype('float32') / 255.0
        faces -= 0.5
        faces *= 2.0

        pred = model.predict(faces).ravel()

        for k, v in labels.items():
            item[v] = float(pred[k])
        
    js = json.dumps(item, indent=2)

    with open(f1, "w") as FOUT:
        FOUT.write(js)


if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest="data/images/",
        save_dest="data/FER_score/",
        new_extension="json",
        old_extension="jpg",
        shuffle=True,
    )(compute, 1)
