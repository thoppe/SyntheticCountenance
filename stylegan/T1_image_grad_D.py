import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob


#f_img = "samples/images/00000042.jpg"
f_img = "movie/alpha_noise_0110/00000110.jpg"



from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import tensorflow as tf
import scipy.stats

def RGB_to_GAN_output(img, resize=True, batch_size=1):
    img = img.transpose(2, 0, 1).astype(float)
    img = 2 * (img / 255.0) - 1
    
    return np.tile(img, (batch_size, 1, 1, 1))


img = np.array(Image.open(f_img))
img = RGB_to_GAN_output(img)
print(img.shape)
#img = np.squeeze(img, axis=0)

G, D, Gs = load_GAN_model()
sess = tf.get_default_session()
tf_img = tf.Variable(img, dtype=tf.float32)

D_scores = D.get_output_for(tf_img, None, is_training=False)
loss = -tf.reduce_sum(D_scores)
grad = tf.gradients(loss, tf_img)[0]

print(D_scores)
print(grad)

sess.run(tf.initializers.variables([tf_img,]))

gx = sess.run(grad, feed_dict={tf_img:img})

np.save("gx.npy", gx)
print(gx)
print(gx.shape)
#######################################################################


# Remove batch dimension
gx = np.load("gx.npy").squeeze()

# Sum over all channels
gx = gx.mean(axis=0)

import pixelhouse as ph

mask = gx<0
img = ph.load(f_img)
img[mask] = 0

img.show()






'''
import seaborn as sns
import pylab as plt

sns.distplot(gx.ravel())
plt.show()
print(gx.shape)
'''
