"""
The idea is to find the img such that D(img) is minimized, that is the picture
that _most_ fools the discriminator.
"""

import numpy as np
from PIL import Image
import PIL

import os, json, glob, random, h5py
from tqdm import tqdm
import tensorflow as tf
from src.GAN_model import load_GAN_model, generate_single
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output

latent_dim = 512
small_image_size = 256

class Image2Latent:
    def __init__(
        self,
        image_dim=small_image_size,
        learning_rate=0.001,
        batch_size=None,
        sess=None,
    ):
        
        model_save_dest = 'model/img2latent'
        os.system(f"mkdir -p {model_save_dest}")

        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

        # Lazy load
        self.generator = None

        input_img = tf.placeholder(
            shape=[None, image_dim, image_dim, 3], dtype=tf.float32)

        z_target = tf.placeholder(
            shape=[None, latent_dim], dtype=tf.float32)

        conv_args = {
            "activation": tf.nn.tanh,
            "kernel_initializer":"he_normal",
        }

        conv = tf.layers.conv2d(input_img, 32, 5, **conv_args)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.dropout(inputs=conv, rate=0.1)

        conv = tf.layers.conv2d(conv, 32, 3,  **conv_args)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.dropout(inputs=conv, rate=0.1)

        conv = tf.layers.conv2d(conv, 32, 3,  **conv_args)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.dropout(inputs=conv, rate=0.1)

        conv = tf.layers.conv2d(conv, 32, 3,  **conv_args)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        conv = tf.layers.dropout(inputs=conv, rate=0.1)        

        fc1 = tf.contrib.layers.flatten(conv)
        fc1 = tf.layers.dense(fc1, latent_dim)

        self.z_output = fc1

        self.loss = tf.reduce_sum(tf.pow(fc1 - z_target, 2))
        self.loss /= image_dim**2
        self.loss /= batch_size
        
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = opt.minimize(self.loss)

        self.model_in = input_img
        self.model_out = z_target

        self.initialize()
        self.saver = tf.train.Saver()
        
    def initialize(self):
        self.sess.run(tf.initializers.global_variables())

    def save(self):
        return self.saver.save(
            self.sess,
            os.path.join(model_save_dest, 'model.ckpt')
        )


    def render(self, z_true, img_true, f_save=None):
        """
        Renders the current latent vector into an image.
        """
        if self.generator is None:
            G, D, Gs = load_GAN_model(sess=self.sess)
            self.generator = Gs
            
        z_test = self.sess.run(
            self.z_output,
            feed_dict = {
                self.model_in : img_true[:1],
            })
        z_test = z_test[0]
        
        img1,_,_ = generate_single(
            self.generator, None, z=z_test, compute_discriminator=False)
        img1 = PIL.Image.fromarray(img1).resize((256,256), Image.ANTIALIAS)

        img2 = img_true[0]
        img2 = np.clip(np.rint((img2 + 1.0) / 2.0 * 255.0),
                       0.0, 255.0).astype(np.uint8)
        img2 = PIL.Image.fromarray(img2)
        
        imgC = np.hstack([img1, img2])
        imgC = PIL.Image.fromarray(imgC)
        
        f_save = 'model/img2latent/running_demo.jpg'
        imgC.save(f_save)


    def train(self, latent_vectors, images):
        """
        """
        lx, _ = self.sess.run(
            [self.loss, self.train_op],
            feed_dict = {
                self.model_in : images,
                self.model_out : latent_vectors,
            })
        return lx

def image_pipeline(batch_size=5):

    f_h5 = 'samples/PGAN_small_images.h5'
    with h5py.File(f_h5, 'r') as h5:
        N = len(h5['Z'])
        Z = h5['Z'][...]

        while True:
            idx = np.random.randint(0, N, size=batch_size)
            img = np.array([h5['IMG'][i] for i in idx])

            img = RGB_to_GAN_output(img, batch_size=batch_size, resize=False)
            yield Z[idx], img



if __name__ == "__main__":
    batch_size = 32
    n_epochs = 2000
    n_save_every = 50

    ITR = image_pipeline(batch_size)
    clf = Image2Latent(batch_size=batch_size)

    while True:

        for n, (z,img) in enumerate(ITR):

            if n%n_save_every == 0:
                clf.render(z, img)
                s = clf.save()

            lx = clf.train(z, img)
            print(n, lx)
