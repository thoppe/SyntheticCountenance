import numpy as np

import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
import pixelhouse as ph

################################################################
from model.memnet.assessors.aestheticsnet import AestheticsNet
import model.memnet.utils.tensorflow

from src.GAN_model import load_GAN_model, logger, generate_single
from src.GAN_model import RGB_to_GAN_output


known_keys = (
    "Aesthetic",
    "BalancingElement",
    "ColorHarmony",
    "Content",
    "DoF",
    "Light",
    "MotionBlur",
    "Object",
    "Repetition",
    "RuleOfThrids",
    "Symmetry",
    "VividColor",
)

target_attribute = "Aesthetic"
direction = 1

# target_attribute = "VividColor"
# direction =1

# target_attribute = "Repetition"
# direction = -1

noise_levels = 0

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


class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01, z_init=None):

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
        img = tf.image.resize_images(img, size=(256, 256))

        # Not sure which one is correct?
        img = tf.transpose(img, [0, 1, 2, 3])

        # RGB to BGR
        # img = img[..., ::-1]

        # Convert to [0, 255]
        img = (tf.clip_by_value(img, -1, 1) + 1) * (255.0 / 2)

        with tf.variable_scope("measure") as scope:

            # Input layer
            input_shape = (256, 256, 3)
            image_input = tf.keras.Input(shape=input_shape)
            model = AestheticsNet()

        score = model.aestheticsnet_fn(
            model.aestheticsnet_preprocess(img), attribute=target_attribute
        )

        self.loss = -tf.reduce_sum(score) * direction

        # Identify the noise vectors
        self.noise_vars = [
            var
            for name, var in Gs.components.synthesis.vars.items()
            if name.startswith("noise")
        ]

        print([x for x in self.noise_vars])
        self.noise_vars = self.noise_vars[:noise_levels]

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        minimize_vars = [self.z]
        if noise_levels > 0:
            minimize_vars = [self.z, self.noise_vars]

        self.train_op = self.opt.minimize(self.loss, var_list=minimize_vars)

        # Convert the generator output into an image
        self.image_output = tflib.convert_images_to_uint8(G_out, nchw_to_nhwc=True)

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

        outputs = [self.loss, self.z, self.train_op]

        feed_dict = {}

        lx, current_latent, _ = self.sess.run(outputs, feed_dict=feed_dict)

        return lx, current_latent, None


if __name__ == "__main__":

    target_idx = 2448
    target_idx = 2401
    target_idx = 2402

    f_z = f"data/latent_vectors/{target_idx:08d}.npy"
    z = np.load(f_z)

    save_dest = f"data/refine_image/{target_idx:08d}"

    os.system(f"rm -rvf {save_dest}")
    os.system(f"mkdir -p {save_dest}")

    learning_rate = 0.025
    # learning_rate = 0.20
    n_save_every = 1

    G, D, Gs = load_GAN_model()
    # Gs = None
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
