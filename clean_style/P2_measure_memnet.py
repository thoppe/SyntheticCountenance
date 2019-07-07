'''
https://github.com/tyshiwo/MemNet

 ICCV'17 paper "MemNet: A Persistent Memory Network for Image Restoration"
 (SPOTLIGHT Presentation)
'''

from src import pipeline
from src.logger import logger
import numpy as np
import json

from model.memnet.assessors.memnet import MemNet
import model.memnet.utils.tensorflow
import tensorflow as tf
import pixelhouse as ph
from tqdm import tqdm


# Input layer
input_shape = (3, 256, 256)
image_input = tf.keras.Input(shape=input_shape)
model = MemNet()

clf = model.memnet_fn(model.memnet_preprocess(image_input))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def compute(f_img, f1):

    img = ph.load(f_img).resize(0.25).rgb
    imgs = np.array([img])
    imgs = np.transpose(imgs, [0, 3, 1, 2])
    res = sess.run(clf, feed_dict={image_input: imgs})
    val = float(res[0][0])

    js = {"f_img": f_img, "memnet_score": val}
    js = json.dumps(js, indent=2)

    with open(f1, "w") as FOUT:
        FOUT.write(js)


if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest="data/images/",
        save_dest="data/MEMNET_score/",
        new_extension="json",
        old_extension="jpg",
        shuffle=True,
    )(compute, 1)
