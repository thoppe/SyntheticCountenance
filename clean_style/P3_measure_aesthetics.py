from src import pipeline
from src.logger import logger
import numpy as np
import json

from model.memnet.assessors.aestheticsnet import AestheticsNet
import model.memnet.utils.tensorflow
import tensorflow as tf
import pixelhouse as ph
from tqdm import tqdm


# Input layer
input_shape = (256, 256, 3)
image_input = tf.keras.Input(shape=input_shape)
model = AestheticsNet()
clf = model.aestheticsnet_fn(model.aestheticsnet_preprocess(image_input))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def compute(f_img, f1):

    img = ph.load(f_img).resize(0.25).rgb
    imgs = np.array([img])
    # imgs = np.transpose(imgs, [0, 3, 1, 2])
    # print(imgs.shape)
    # exit()
    res = sess.run(clf, feed_dict={image_input: imgs})

    for key, val in res.items():
        res[key] = float(val[0][0])

    res["f_img"] = f_img
    js = json.dumps(res, indent=2)

    with open(f1, "w") as FOUT:
        FOUT.write(js)


# https://github.com/aimerykong/deepImageAestheticsAnalysis
# Source for the original model:
# Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016). Photo aesthetics ranking network with attributes and content adaptation. ArXiv CS, 1606.01621. Retrieved from http://arxiv.org/abs/1606.01621

if __name__ == "__main__":
    PIPE = pipeline.Pipeline(
        load_dest="data/images/",
        save_dest="data/Aesthetics_score/",
        new_extension="json",
        old_extension="jpg",
        shuffle=True,
    )(compute, 1)
