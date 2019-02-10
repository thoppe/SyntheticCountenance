"""
NEEDS WORK.

GAN inverter

Usage:
  T0_match_image.py <image_file> [--learning_rate=<f>] [--restart]

Options:
  -h --help     Show this screen.
  --learning_rate=<f>  ADAM learning rate [default: 0.05]
  --restart
"""

"""
Training 00000042.jpg on 250 epochs, loss:
1] Single z: 0.6198  (2.2k npy files)
2] Multi z (10/18 layers): 0.3794 (2.2k npy file, but looks bad!?)
3] Single z (with noise):
4] Multi z (with noise):
"""

from docopt import docopt

#import dlib
import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
from src.GAN_model import load_GAN_model, logger, generate_single
from src.GAN_model import RGB_to_GAN_output

import cv2
import sklearn.decomposition

np.random.seed(48)
n_image_upscale = 1

#model_dest = "model/dlib"
#shape68_pred = dlib.shape_predictor(
#    f"{model_dest}/shape_predictor_68_face_landmarks.dat"
#)
#detector = dlib.get_frontal_face_detector()

class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01,
                 use_mask=True, use_multi=True, z_init=None):

        self.sess = sess
        self.target_image = None
        self.use_mask = use_mask
        self.use_multi = use_multi

        latent_dim = 512
        image_dim = 1024
        batch_size = 1
        
        # Single matching
        if not self.use_multi:
            z_init = np.random.randn(latent_dim)[None, :]
            self.z = tf.Variable(z_init, dtype=tf.float32)
            G_out = generator.get_output_for(self.z, None, is_training=False)
            
        # Multi matching
        elif self.use_multi:

            n_opt = 10
            z_init0 = np.random.standard_normal(size=[n_opt, 512])
            z_init1 = np.random.standard_normal(size=[18-n_opt, 512])
                       
            self.z = tf.Variable(z_init0, dtype=tf.float32)
            self.z2 = tf.Variable(z_init1, dtype=tf.float32, trainable=False)

            # The first one fails badly
            #z_val = tf.concat([self.z, self.z2], 0)
            z_val = tf.concat([self.z2, self.z], 0)

            G_out = Gs.components.synthesis.get_output_for(
                [z_val],
                is_validation=True, randomize_noise=False,
            )

        # NCHW
        self.img_in = tf.placeholder(
            tf.float32, shape=(batch_size, 3, image_dim, image_dim)
        )

        # L1 error, only train the loss
        L1_loss = tf.abs(G_out - self.img_in)

        # Sum over the batch_size, channel info
        L1_loss = tf.reduce_sum(L1_loss, axis=0)
        L1_loss = tf.reduce_sum(L1_loss, axis=0)

        # Sum over all pixels
        #if use_mask:
        #    L1_loss *= self.mask
        self.loss = tf.reduce_sum(L1_loss ** 2)
        self.loss /= 1024**2
        #self.loss /= tf.reduce_sum(self.mask)

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #self.opt = tf.train.GradientDescentOptimizer(
        #    learning_rate=learning_rate)

        # Only train the latent variable (hold the generator fixed!)
        minimize_vars = [self.z,]
        
        self.train_op = self.opt.minimize(
            self.loss, var_list=minimize_vars)

    def initialize(self):
        if self.use_multi:
            self.sess.run(tf.initializers.variables([self.z, self.z2]))
        else:
            self.sess.run(tf.initializers.variables([self.z]))
        
        self.sess.run(tf.initializers.variables(self.opt.variables()))

    def set_target(self, f_image):
        """
        For now, can only load from a file.
        """
        
        # Load the target image
        img = Image.open(f_image)
        self.target_image = RGB_to_GAN_output(img)

    def render(self, f_save=None):
        """
        Renders the current latent vector into an image.
        """
        tf_latent = self.z
        
        current_latent = self.sess.run(
            tf_latent,
            feed_dict={
                self.img_in: self.target_image,
                #self.mask: self.target_mask
            },
        )
        
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        img = Gs.run(
            current_latent,
            None,
            truncation_psi=0.7,
            use_noise=False,
            randomize_noise=False,
            output_transform=fmt
        )[0]

        if f_save is not None:
            P_img = Image.fromarray(img)
            P_img.save(f_save)

            f_npy = f_save.replace(".jpg", ".npy")
            np.save(f_npy, current_latent)

        return img

    def train(self):
        """
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        """

        if self.target_image is None:
            raise ValueError("must call .set_target(img) first!")
                
        tf_latent = self.z

        outputs = [self.loss, tf_latent, self.train_op]
        lx, current_latent, _ = self.sess.run(
            outputs,
            feed_dict={
                self.img_in: self.target_image,
                #self.mask: self.target_mask
            },
        )

        return lx, current_latent



if __name__ == "__main__":

    cargs = docopt(__doc__, version='GAN inverter 0.1')
    f_image = cargs['<image_file>']

    if not os.path.exists(f_image):
        logger.error(f"File not found, {f_image}")
        raise ValueError

    name = os.path.basename(f_image).split('.')[0]

    n_save_every = 10
    is_restart = False
    learning_rate = float(cargs['--learning_rate'])

    save_dest = f"samples/match_image/{name}"
    os.system(f'mkdir -p {save_dest}')

    f_processed = os.path.join(
        "samples/match_image/",
        f"match_{name}.jpg"
    )

    # Random init fool!
    #z_init = np.random.randn(512)

    G, D, Gs = load_GAN_model()

    
    sess = tf.get_default_session()
    
    GI = GeneratorInverse(Gs, sess, learning_rate=learning_rate, z_init=None)
    GI.initialize()

    logger.info(f"Starting training against {f_image}")
    GI.set_target(f_image)

    for i in range(0, 20000):

        # Only save every 10 iterations
        if i % n_save_every == 0:
            GI.render(f"{save_dest}/{i:05d}.jpg")

        loss, z = GI.train()
        norm = np.linalg.norm(z) / np.sqrt(512)
        print(z.shape)
        logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")
