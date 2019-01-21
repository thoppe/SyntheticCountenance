import tensorflow as tf
import numpy as np
import os
from PIL import Image
import PIL

from .GAN_model import load_GAN_model, generate_single

small_image_size = 256
latent_dim = 512

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

    def __call__(self, img):
        '''
        Given a properly preprocesssed image, 
        return the best fit for a latent vector.
        '''
        return self.sess.run(
            self.z_output,
            feed_dict = {
                self.model_in : [img,],
        })[0]

    def draw_latent(self, z):
        
        if self.generator is None:
            G, D, Gs = load_GAN_model(sess=self.sess)
            self.generator = Gs

        img1,_,_ = generate_single(
            self.generator, None, z=z, compute_discriminator=False)
        img1 = PIL.Image.fromarray(img1)

        return img1
        


    def render(self, z_true, img_true, f_save=None):
        """
        Renders the current latent vector into an image.
        """

        z_test = self(img_true[0])

        img1 = self.draw_latent(self(img_true[0]))
        img1 = img1.resize((256,256), Image.ANTIALIAS)

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
