"""
The idea is to find the img such that D(img) is minimized, that is the picture
that _most_ fools the discriminator.
"""

import numpy as np
from PIL import Image

import os, json, glob, random
from tqdm import tqdm
import tensorflow as tf
from src.GAN_model import load_GAN_model
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output

latent_dim = 512

class Image2Latent:
    def __init__(
        self,
        image_dim=1024//4,
        learning_rate=0.001,
    ):

        self.sess = tf.InteractiveSession()

        input_img = tf.placeholder(
            shape=[None, image_dim, image_dim, 3], dtype=tf.float32)

        z_target = tf.placeholder(
            shape=[None, latent_dim], dtype=tf.float32)

        conv1 = tf.layers.conv2d(input_img, 32, 5, activation=tf.nn.tanh)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.tanh)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        conv3 = tf.layers.conv2d(conv2, 32, 3, activation=tf.nn.tanh)
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc1, latent_dim)

        self.loss = tf.reduce_sum(tf.pow(fc1 - z_target, 2))
        self.loss /= image_dim**2
        
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = opt.minimize(self.loss)

        self.model_in = input_img
        self.model_out = z_target

        self.initialize()
        
    def initialize(self):
        self.sess.run(tf.initializers.global_variables())

    def render(self, f_save=None):
        """
        Renders the current latent vector into an image.
        """
        raise NotImplemented

    def train(self, latent_vectors, images):
        """
        sdklfjsldf
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        """
        lx, _ = self.sess.run(
            [self.loss, self.train_op],
            feed_dict = {
                self.model_out : latent_vectors,
                self.model_in : images,
            })
        return lx

def image_pipeline(batch_size=5):
    F_LATENT = glob.glob('samples/latent_vectors/*.npy')
    while True:

        f_z = random.sample(F_LATENT, batch_size)
        z = np.array([np.load(f) for f in f_z])
        f_img = [
            os.path.join('samples', 'images',
                         os.path.basename(f).replace('.npy', '.jpg'))
            for f in f_z]

        if not all(map(os.path.exists, f_img)):
            continue
        
        #img = [Image.open(f).resize((256,256), Image.ANTIALIAS) for f in f_img]
        img = [Image.open(f).resize((256,256)) for f in f_img]

        
        img = [RGB_to_GAN_output(np.array(x), resize=False) for x in img]
        img = np.array(img)

        yield z, img



batch_size = 16

ITR = image_pipeline(batch_size)
clf = Image2Latent()

for k, (z,img) in enumerate(ITR):
    lx = clf.train(z, img)
    print(k, lx)

exit()

# np.random.seed(45)
#G, D, Gs, sess = load_GAN_model(return_sess=True)
#MD = MaximizeDiscriminator(
#    D, sess, f_img=f_img, learning_rate=0.01, find_most_likely=True, blend=0.00001
#)

MD.initialize()

print("Starting training")
save_dest = "samples/maximizer_demo"

os.system(f"rm -rvf {save_dest} && mkdir -p {save_dest}")

for i in range(200_000):

    # Only save every 10 iterations
    if i % 10 == 0:

        # Doesn't work
        # if i:
        #    # Try to color match to the original each time
        #    img1 = GAN_output_to_RGB(gimg)[0]
        #    img2 = color_transfer(img0, img1)
        #    gimg2 = RGB_to_GAN_output(img2)
        #    MD.img.load(gimg2, sess)

        MD.render(f"{save_dest}/{i:05d}.jpg")

    loss, gimg = MD.train()
    print(f"Epoch {i}, loss {loss:0.4f}")
