import tensorflow as tf
import numpy as np
from assessors.memnet import MemNet
import pixelhouse as ph
import utils.tensorflow
from tqdm import tqdm

clf = MemNet()

# works
f_mean = "assessors/memnet_mean.mat"
print(clf.get_mu(f_mean).shape)

img0 = ph.load("img_68.jpg").rgb
img1 = ph.load("img_89.jpg").rgb

print(img0.shape)
#img0 = img0[:, :, ::-1]
#img1 = img1[:, :, ::-1]


imgs = np.array([img0, img1])

# Swap channels
imgs = np.transpose(imgs, [0, 3, 1, 2])

# works
#f_weights = 'assessors/memnet_state_dict.p'
#state_dict = clf.get_weights(f_weights)

# Channel first BGR image
input_shape = (3, 256, 256)

# Input layer
image_input = tf.keras.Input(shape=input_shape)

imgx = clf.memnet_preprocess(image_input)
imgy = clf.memnet_fn(imgx)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for n in tqdm(range(1000)):
    res = sess.run(imgy, feed_dict={image_input:imgs})
    print(res)
