import numpy as np

import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
import pixelhouse as ph


################################################################
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run(
        [tf.is_variable_initialized(var) for var in global_vars]
    )
    not_initialized_vars = [
        v for (v, f) in zip(global_vars, is_not_initialized) if not f
    ]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
        
def preprocess_image_for_VGG(f_img, IMG_SIZE=160):
    # Takes in a filename returns a [1, 160, 160, 3] tensor
    img = tf.read_file(f_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = (img/127.5) - 1
    img = tf.image.resize_images(img, size=(IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img,0)
    return img

################################################################
# 1. First figure out how to get VGG features from an image (DONE)
# 2. Get feature maps from generated image (DONE)
# 3. Craft loss function by trying them together
# 4. REFINE BITACH
################################################################

'''
# 1. First figure out how to get VGG features from an image

tf.enable_eager_execution()

IMG_SIZE = 160

VG_IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG=tf.keras.applications.VGG16(
    input_shape=VG_IMG_SHAPE, include_top=False, weights='imagenet')
VGG.trainable=False

f_img = 'pixelhouseImage_screenshot_03.07.2019.png'

img = tf.read_file(f_img)
img = tf.image.decode_jpeg(img, channels=3)
img = preprocess_image_for_VGG(img)
VGG_pool = tf.keras.layers.GlobalAveragePooling2D()

vg = VGG(tf.expand_dims(img,0))
zg = tf.squeeze(VGG_pool(vg))
'''


################################################################

from src.GAN_model import load_GAN_model, logger, generate_single
from src.GAN_model import RGB_to_GAN_output

noise_levels = 4

class GeneratorInverse:
    def __init__(
            self, generator, sess, learning_rate=0.01, z_init=None,
            f_target_image=None,
    ):

        self.sess = sess
        self.canvas = ph.Canvas()

        # Resize the latent vector
        z_init = z_init[np.newaxis, :]
        self.z = tf.Variable(z_init, dtype=tf.float32)

        # Identify the noise vectors
        self.noise_vars = [
            var
            for name, var in Gs.components.synthesis.vars.items()
            if name.startswith("noise")
        ]

        # Find the output to the generator
        G_out = generator.get_output_for(
            self.z,
            None,
            is_training=False,
            randomize_noise=False,
            use_noise=True,
            structure="fixed",
        )
        img = G_out

        # Not sure which one is correct?
        img = tf.transpose(img, [0, 3, 2, 1])

        # Resize the image
        img = tf.image.resize_images(img, size=(160, 160))

        # Not sure which one is correct?
        img0 = tf.transpose(img, [0, 1, 2, 3])


        ### VGG layers FUNCTIONS
        # This may not be properly preprocessing!
        with tf.variable_scope("measure") as scope:
            IMG_SIZE = 160
            VG_IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
            VGG=tf.keras.applications.VGG16(
                input_shape=VG_IMG_SHAPE, include_top=False, weights='imagenet')
            VGG.trainable=False

            VGG_pool = tf.keras.layers.GlobalAveragePooling2D()
            v0 = tf.squeeze(VGG_pool(VGG(img)))

            img1 = preprocess_image_for_VGG(f_target_image)
            v1 = tf.squeeze(VGG_pool(VGG(img1)))

        self.loss_VGG = 100 * tf.reduce_mean(tf.abs(v0-v1)**2)
        self.loss_pixel = 0*tf.reduce_mean(tf.abs(img0 - img1)**2)

        self.loss = self.loss_VGG + self.loss_pixel
        #self.loss = -tf.reduce_sum(score) * direction

        # Identify the noise vectors
        self.noise_vars = [
            var
            for name, var in Gs.components.synthesis.vars.items()
            if name.startswith("noise")
        ]


        self.noise_vars = self.noise_vars[:noise_levels]
        print(self.noise_vars)

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        minimize_vars = [self.z]
        if noise_levels > 0:
            minimize_vars = [self.z, self.noise_vars]

        self.train_op = self.opt.minimize(self.loss, var_list=minimize_vars)

        # Convert the generator output into an image
        self.image_output = tflib.convert_images_to_uint8(
            G_out, nchw_to_nhwc=True)

        ############################################################
        self.initialize()

    def initialize(self):
        self.sess.run(tf.initializers.variables([self.z]))
        self.sess.run(tf.initializers.variables(self.opt.variables()))
        self.sess.run(tf.initializers.variables(self.noise_vars))
        
        # Need to keep the networks separate
        initialize_uninitialized(self.sess)

    def render(self):
        img = self.sess.run(self.image_output)[0]
        self.canvas.img = img
        self.canvas.show(1)

    def train(self):
        """
        For each training step ...
        """

        tf_latent = self.z

        outputs = [self.loss_pixel, self.loss_VGG, self.z, self.train_op]
        feed_dict = {}

        lx_px, lx_VGG, current_latent, _ = self.sess.run(
            outputs, feed_dict=feed_dict)

        return lx_px, lx_VGG, current_latent, None


if __name__ == "__main__":

    target_idx = 2448
    #target_idx = 2401
    #target_idx = 2402

    f_fit = f"data/images/00000159.jpg"

    f_z = f"data/latent_vectors/{target_idx:08d}.npy"
    z = np.load(f_z)

    save_dest = f"data/FIT_IMAGE/{target_idx:08d}"

    os.system(f"rm -rvf {save_dest}")
    os.system(f"mkdir -p {save_dest}")

    learning_rate = 0.05
    n_save_every = 1

    G, D, Gs = load_GAN_model()
    sess = tf.get_default_session()

    GI = GeneratorInverse(
        Gs, sess, learning_rate=learning_rate, z_init=z,
        f_target_image=f_fit,
    )
    GI.initialize()

    logger.info(f"Starting training against {target_idx}")

    for i in range(0, 200000):

        # Only save every 10 iterations
        #if i % n_save_every == 0:
        #    GI.render()
        GI.render()

        lx_px, lx_VGG, z, _ = GI.train()
        norm = np.linalg.norm(z) / np.sqrt(512)

        logger.debug(
            f"Epoch {i}, pixel loss {lx_px:0.4f}, "
            f"VGG loss {lx_VGG:0.4f} z-norm {norm:0.4f} "
        )
