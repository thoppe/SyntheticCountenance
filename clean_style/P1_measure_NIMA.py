'''
Adapted from:

NIMA: Neural Image Assessment
Implementation of NIMA: Neural Image Assessment in Keras + Tensorflow with weights for MobileNet model trained on AVA dataset.

https://github.com/titu1994/neural-image-assessment
'''

from src import pipeline
import json, os
import numpy as np

from statsmodels.stats.weightstats import DescrStatsW

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

use_GPU = True
device = "/GPU:0" if use_GPU else "/CPU:0"

with tf.device(device):
    base_model = MobileNet(
        (None, None, 3), alpha=1, include_top=False, pooling="avg", weights=None
    )
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation="softmax")(x)
    model = Model(base_model.input, x)
    model.load_weights("src/mobilenet_weights.h5")


def process_image(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    scores = model.predict(x, batch_size=1, verbose=0)[0]
    return scores


def compute(f0, f1):
    
    target_size = (224, 224)
    #target_size = None

    img = load_img(f0, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    scores = model.predict(x, batch_size=1, verbose=0)[0]

    weighted_stats = DescrStatsW(range(len(scores)), weights=scores, ddof=0)

    js = {
        "f_img": f0,
        "NIMA_mean": float(weighted_stats.mean),
        "NIMA_std": float(weighted_stats.std),
    }
    js = json.dumps(js, indent=2)
    print(js)

    with open(f1, "w") as FOUT:
        FOUT.write(js)


if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest="data/images/",
        save_dest="data/NIMA_score/",
        new_extension="json",
        old_extension="jpg",
        shuffle=True,
    )(compute, 1)
