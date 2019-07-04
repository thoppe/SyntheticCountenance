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
        self.img_out = G_out

        # Convert the generator output into an image
        self.image_output = tflib.convert_images_to_uint8(
            self.img_out, nchw_to_nhwc=True)

        # Preprocess the image
        print(self.image_output)

        img = tf.cast(self.image_output,  dtype=tf.float32)
        print(img)
        
        img /= 127.5
        img -= 1.0
        score = AES.predict(img, steps=1, verbose=0)
        
        print(score)
        exit()
        



        ############################################################
        self.initialize()        
        img = self.sess.run(self.image_output)[0]

        print(img)

        cv = ph.Canvas()
        cv.img = img
        cv.show()
        
        print(img.shape)
        print(aesthetic_model(img))
        
        exit()

        exit()

        '''
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


        self.noise_loss = 0
        for nx in self.noise_vars:
            noise_norm = tf.sqrt(tf.reduce_sum(nx**2))
            expected_norm = tf.sqrt(tf.reduce_prod(tf.to_float(tf.shape(nx))))

            self.noise_loss += (noise_norm - expected_norm) ** 2

        '''
        
        self.img_out = G_out

        # NCHW
        self.img_in = tf.placeholder(
            tf.float32, shape=(batch_size, 3, image_dim, image_dim)
        )

        # L1 error, only train the loss
        #L1_loss = tf.pow(G_out - self.img_in, 2)
        L1_loss = tf.abs(G_out - self.img_in)

        # Sum over the batch_size, channel info
        L1_loss = tf.reduce_sum(L1_loss, axis=0)
        L1_loss = tf.reduce_sum(L1_loss, axis=0)

        # Sum over all pixels
        self.tf_mask = tf.placeholder(dtype=tf.float32)
        L1_loss *= self.tf_mask
        
        self.loss = tf.reduce_sum(L1_loss ** 2)
        self.loss /= tf.reduce_sum(self.tf_mask)

        # Add the noise variance loss
        self.noise_loss /= len(self.noise_vars)
        self.noise_loss *= 0.001
        
        self.loss += self.noise_loss


        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #self.opt = tf.train.GradientDescentOptimizer(
        #    learning_rate=learning_rate)
        
        minimize_vars = [self.z, self.noise_vars]
        
        self.train_op = self.opt.minimize(
            self.loss, var_list=minimize_vars)

        self.image_output = tflib.convert_images_to_uint8(
            self.img_out, nchw_to_nhwc=True)

    def initialize(self):
        
        self.sess.run(tf.initializers.variables([self.z]))
        self.sess.run([self.noise_vars])

        # FIX LATER
        #self.sess.run(tf.initializers.variables(self.opt.variables()))
        

    def render(self, f_save=None):
        """
        Renders the current latent vector into an image.
        """
        tf_latent = self.z
        feed_dict={
            self.img_in: self.target_image,
            self.tf_mask: self.mask
        }

        current_latent = self.sess.run(tf_latent, feed_dict=feed_dict)
        noise = self.sess.run(self.noise_vars, feed_dict=feed_dict)
                
        img = self.sess.run(self.image_output)[0]

        
        if f_save is not None:
            P_img = Image.fromarray(img)
            P_img.save(f_save)

            #f_npy = f_save.replace(".jpg", ".npy")
            #np.save(f_npy, current_latent)

            f_npz = f_save.replace(".jpg", ".npz")
            kwargs = {"z":current_latent}
            norms = []
            for k, nx in enumerate(noise):
                kwargs[f'noise_{k}'] = nx
                x = nx.ravel()
                norms.append(np.linalg.norm(x)/np.sqrt(len(x)))
            print(np.array(norms))
            np.savez(f_npz, **kwargs)

            

        return img

    def train(self):
        """
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        """

        if self.target_image is None:
            raise ValueError("must call .set_target(img) first!")
                
        tf_latent = self.z

        outputs = [self.loss, tf_latent, self.noise_loss, self.train_op,]
        lx, current_latent, nx, _ = self.sess.run(
            outputs,
            feed_dict={
                self.img_in: self.target_image,
                self.tf_mask: self.mask,
            },
        )

        return lx, current_latent, nx



if __name__ == "__main__":


    # KNOWN LOSS is 3.19
    target_idx = 2448 
    f_z = f"data/latent_vectors/{target_idx:08d}.npy"
    z = np.load(f_z)

    save_dest = f"data/refine_image/{target_idx:08d}"

    os.system(f'rm -rvf {save_dest}')
    os.system(f'mkdir -p {save_dest}')

    is_restart = False
    learning_rate = 0.025

    G, D, Gs = load_GAN_model()
    #G, D, Gs = [None,]*3
    sess = tf.get_default_session()
    
    GI = GeneratorInverse(Gs, sess, learning_rate=learning_rate, z_init=z)
    GI.initialize()

    exit()
    
    logger.info(f"Starting training against {f_image}")
    GI.set_target(f_image)

    for i in range(0, 200000):

        # Only save every 10 iterations
        if i % n_save_every == 0:
            GI.render(f"{save_dest}/{i:05d}.jpg")

        loss, z, noise_loss = GI.train()
        norm = np.linalg.norm(z) / np.sqrt(512)
        #logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")
        logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f} "
                     f"noise-loss {noise_loss:04f}")
