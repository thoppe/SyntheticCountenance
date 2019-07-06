import numpy as np

import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
from src.GAN_model import load_GAN_model, logger, generate_single
from src.GAN_model import RGB_to_GAN_output

import pixelhouse as ph

################################################################

from model.memnet.assessors.memnet import MemNet
import model.memnet.utils.tensorflow

################################################################

class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01,
                 z_init=None):

        self.sess = sess
        self.canvas = ph.Canvas()

        # Load the memnet model
        input_shape = (3, 256, 256)
        image_input = tf.keras.Input(shape=input_shape)
        model = MemNet()


        # Resize the latent vector
        z_init = z_init[np.newaxis, :]
        self.z = tf.Variable(z_init, dtype=tf.float32)

        # Identify the noise vectors
        self.noise_vars = [
            var for name, var in Gs.components.synthesis.vars.items()
            if name.startswith('noise')
        ]

        # Find the output to the generator
        G_out = generator.get_output_for(
            self.z, None,
            is_training=False,
            randomize_noise=False,
            use_noise=True,
            structure='fixed',
        )
        img = G_out

        # Not sure which one is correct?
        img = tf.transpose(img, [0, 3, 2, 1])

        print(img)

        # Resize the image
        img = tf.image.resize_images(img, size=(256,256))

        # RGB to BGR
        img = img[..., ::-1]

        # Convert to [0, 255]
        img = (tf.clip_by_value(img, -1, 1) + 1)*(255.0/2)

        score = model.memnet_fn(model.memnet_preprocess(img))
        
        self.loss = -tf.reduce_sum(score)
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        minimize_vars = [self.z, ]
        #minimize_vars = [self.z, self.noise_vars[:12]]
        
        self.train_op = self.opt.minimize(
            self.loss, var_list=minimize_vars)

        # Convert the generator output into an image
        self.image_output = tflib.convert_images_to_uint8(
            G_out, nchw_to_nhwc=True)

        ############################################################
        self.initialize()

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.initializers.variables([self.z]))
        self.sess.run(tf.initializers.variables(self.opt.variables()))
        
        #self.sess.run([self.noise_vars])

    def render(self):
        img = self.sess.run(self.image_output)[0]
        self.canvas.img = img
        self.canvas.show(1)

    def train(self):
        """
        For each training step ...
        """
                
        tf_latent = self.z

        outputs = [
            self.loss,
            self.z,
            self.train_op,
        ]

        feed_dict = {}
        
        lx, current_latent, _ = self.sess.run(
            outputs, feed_dict=feed_dict)

        return lx, current_latent, None
    


if __name__ == "__main__":

    target_idx = 2448
    #target_idx = 2401
    
    f_z = f"data/latent_vectors/{target_idx:08d}.npy"
    z = np.load(f_z)

    save_dest = f"data/refine_image/{target_idx:08d}"

    os.system(f'rm -rvf {save_dest}')
    os.system(f'mkdir -p {save_dest}')

    is_restart = False
    learning_rate = 0.025
    #learning_rate = 0.20
    n_save_every = 1

    G, D, Gs = load_GAN_model()
    #Gs = None
    
    sess = tf.get_default_session()
    
    GI = GeneratorInverse(Gs, sess, learning_rate=learning_rate, z_init=z)
    GI.initialize()
    
    logger.info(f"Starting training against {target_idx}")

    for i in range(0, 200000):
        
        # Only save every 10 iterations
        if i % n_save_every == 0:
            GI.render()

        loss, z, _ = GI.train()
        norm = np.linalg.norm(z) / np.sqrt(512)
        
        logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f} ")
