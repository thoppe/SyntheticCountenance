from src import pipeline
from src.logger import logger
import cv2
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

use_GPU = False
device = "/GPU:0" if use_GPU else "/CPU:0"

with tf.device(device):
    base_model = MobileNet((None, None, 3), alpha=1,
                           include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights('src/mobilenet_weights.h5')

def process_image(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    scores = model.predict(x, batch_size=1, verbose=0)[0]
    return scores

    
def compute(f0, f1):
    
    img = load_img(f0, target_size=None)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    scores = model.predict(x, batch_size=1, verbose=0)[0]

    np.save(f1, scores)

    #np.save(f1, pts)

if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest = 'data/images/',
        save_dest = 'data/AS_score/',
        new_extension = 'npy',
        old_extension = 'jpg',
        shuffle=False,
    )(compute, 1)
