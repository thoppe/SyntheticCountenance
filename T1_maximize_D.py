'''
The idea is to find the img such that D(img) is minimized, that is the picture
that _most_ fools the discriminator.
'''

import numpy as np
from PIL import Image

import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output

class MaximizeDiscriminator:

    def __init__(self, discriminator, sess, learning_rate=0.01):

        self.sess = sess

        latent_dim = 512
        image_dim = 1024
        batch_size = 1
        
        # Start with random init for the latents
        img0 = np.random.uniform(
            -1, 1, size=[3, image_dim, image_dim])

        self.img = tf.Variable(
            img0[None, :, :, :], dtype=tf.float32)

        
        D_scores, D_labels = discriminator.get_output_for(
            self.img, is_training=False)

        self.loss = tf.reduce_sum(D_scores)
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.img,])

    def initialize(self):
        self.sess.run(
            tf.initializers.variables([self.img,]))
        
        self.sess.run(
            tf.initializers.variables(self.opt.variables()))

    def set_target(self, f_image):
        '''
        For now, can only load from a file.
        '''        
        np_img = np.array(Image.open(f_img))
        self.target_img = RGB_to_GAN_output(np_img)


    def render(self, f_save=None):
        '''
        Renders the current latent vector into an image.
        '''
        g_img = self.sess.run(self.img)
        img = GAN_output_to_RGB(g_img)[0]

        if f_save is not None:
            P_img = Image.fromarray(img)        
            P_img.save(f_save)

        return img
        
    def train(self):
        '''
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        '''        
        outputs = [self.loss, self.train_op]
        lx,_ = self.sess.run(outputs)
        return lx

f_img = 'samples/images/000360.jpg'
    
np.random.seed(45)
G, D, Gs, sess = load_GAN_model(return_sess=True)
MD = MaximizeDiscriminator(D, sess, learning_rate=0.01)

MD.set_target(f_img)
MD.initialize()

print ("Starting training")
save_dest = 'samples/maximizer_demo'

os.system(f'rm -rvf {save_dest} && mkdir -p {save_dest}')

for i in range(200000):

    # Only save every 10 iterations
    if i%10==0:
        MD.render(f'{save_dest}/{i:05d}.jpg')

    loss = MD.train()
    print(f"Epoch {i}, loss {loss:0.4f}")

    
