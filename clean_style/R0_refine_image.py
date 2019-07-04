import numpy as np

from U0_transform_image import Image_Transformer
import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
from src.GAN_model import load_GAN_model, logger, generate_single
from src.GAN_model import RGB_to_GAN_output

import pixelhouse as ph

################################################################

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

################################################################


class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01,
                 z_init=None):

        self.sess = sess
        self.canvas = ph.Canvas()

        # Load the aesthetic model
        AES_model = MobileNet(
            (None, None, 3), alpha=1,
            include_top=False, pooling='avg', weights=None
        )
        AES = Dropout(0.75)(AES_model.output)
        AES = Dense(10, activation='softmax')(AES)
        AES = Model(AES_model.input, AES)
        AES.load_weights('src/mobilenet_weights.h5')

        
        # Set stylegan parameters
        latent_dim = 512
        image_dim = 1024
        batch_size = 1

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

        # Not sure which one is correct?
        #img = tf.transpose(G_out, [0, 2, 3, 1])
        img = tf.transpose(G_out, [0, 3, 2, 1])

        
        img = tf.clip_by_value(img, -1, 1)
        
        score = AES(img)
        score = tf.squeeze(score)

        self.loss = -tf.reduce_sum(score*[0,1,2,3,4,5,6,7,8,9])        
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        #self.opt = tf.train.GradientDescentOptimizer(
        #    learning_rate=learning_rate)


        #minimize_vars = [self.z, self.noise_vars]
        #minimize_vars = [self.z, ]
        print(len(self.noise_vars))
        minimize_vars = [self.z, self.noise_vars[:10]]
        
        self.train_op = self.opt.minimize(
            self.loss, var_list=minimize_vars)

        # Convert the generator output into an image
        self.image_output = tflib.convert_images_to_uint8(
            G_out, nchw_to_nhwc=True)

        ############################################################
        self.initialize()

        return None



        '''
        self.noise_loss = 0
        for nx in self.noise_vars:
            noise_norm = tf.sqrt(tf.reduce_sum(nx**2))
            expected_norm = tf.sqrt(tf.reduce_prod(tf.to_float(tf.shape(nx))))

            self.noise_loss += (noise_norm - expected_norm) ** 2
        
        self.noise_loss /= len(self.noise_vars)
        self.noise_loss *= 0.001
        self.loss += self.noise_loss
        '''


    def initialize(self):
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


    # KNOWN LOSS is 3.19, currently this doesn't match up!!!
    target_idx = 2448 
    f_z = f"data/latent_vectors/{target_idx:08d}.npy"
    z = np.load(f_z)

    save_dest = f"data/refine_image/{target_idx:08d}"

    os.system(f'rm -rvf {save_dest}')
    os.system(f'mkdir -p {save_dest}')

    is_restart = False
    #learning_rate = 0.025
    learning_rate = 0.050
    n_save_every = 1

    G, D, Gs = load_GAN_model()
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
