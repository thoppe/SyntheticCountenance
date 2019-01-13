import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output

from P3_keypoints_68 import compute_keypoints
from P1_compute_bbox import compute_bbox

import cv2
import scipy.spatial

def compute_convex_hull_face(img):
    ''' Assume img is loaded from cv2.imread '''
    
    bbox = compute_bbox(img)
    assert(len(bbox)==1)
    
    keypoints = compute_keypoints(img, bbox[0])
    hull = cv2.convexHull(keypoints)
    mask = np.zeros(img.shape,np.uint8)

    cv2.drawContours(mask,[hull], 0, 1, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    mask = cv2.dilate(mask, kernel, iterations = 1)  
    mask = mask.astype(bool)[:, :, 0]
    
    return mask


class GeneratorInverse:

    def __init__(
            self, generator, sess, learning_rate=0.01, use_mask=True,
            z_init=None,
    ):

        self.sess = sess
        self.target_image = None
        self.use_mask = use_mask

        latent_dim = 512
        image_dim = 1024
        batch_size = 1

        if z_init is None:
            # Start with random init for the latents
            z_init = np.random.randn(latent_dim)[None, :]
        
        self.z = tf.Variable(z_init, dtype=tf.float32)
        self.mask = tf.placeholder(dtype=tf.float32)
        
        # Labels are not needed for this project
        label_dummy = tf.zeros([batch_size,0])
    
        G_out = generator.get_output_for(
            self.z, label_dummy, is_training=False)

        # NCHW
        self.img_in = tf.placeholder(
            tf.float32, shape=(batch_size, 3, image_dim, image_dim))

        # L1 error, only train the loss
        L1_loss = tf.abs(G_out - self.img_in)

        # Sum over the batch_size, channel info
        L1_loss = tf.reduce_sum(L1_loss, axis=0)
        L1_loss = tf.reduce_sum(L1_loss, axis=0)

        # Sum over all pixels
        if use_mask:
            L1_loss *= self.mask
        
        self.loss = tf.reduce_sum(L1_loss**2)
        self.loss /= tf.reduce_sum(self.mask)
    
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #self.opt = tf.train.GradientDescentOptimizer(
        #learning_rate=learning_rate)

        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.z,])

    def initialize(self):
        self.sess.run(
            tf.initializers.variables([self.z,]))
        
        self.sess.run(
            tf.initializers.variables(self.opt.variables()))

    def set_target(self, f_image):
        '''
        For now, can only load from a file.
        '''

        if self.use_mask:
            # Load a mask (all CV2 operations)
            cv_img = cv2.imread(f_image)
            mask = compute_convex_hull_face(cv_img)
            self.target_mask = np.ones_like(mask)
            
            # Set values off the mask to still be important
            self.target_mask[~mask] = 0.25
        else:
            # Debug line
            self.target_mask = np.ones((1024, 1024))

        # Load the target image
        img = Image.open(f_image)        
        self.target_image = RGB_to_GAN_output(img)


    def render(self, f_save=None):
        '''
        Renders the current latent vector into an image.
        '''
        z_current = self.sess.run(
            self.z, feed_dict={
                self.img_in:self.target_image,
                self.mask:self.target_mask,
            })

        img = Gs.run(z_current, np.zeros([512,0]))
        img = GAN_output_to_RGB(img)[0]

        if f_save is not None:
            P_img = Image.fromarray(img)        
            P_img.save(f_save)

            f_npy = f_save.replace('.jpg', '.npy')
            np.save(f_npy, z_current)

        return img
        
    def train(self):
        '''
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        '''
        
        if self.target_image is None:
            raise ValueError("must call .set_target(img) first!")            
        
        outputs = [self.loss, self.z, self.train_op]
        lx,z,_ = self.sess.run(
            outputs, feed_dict={
                self.img_in:self.target_image,
                self.mask:self.target_mask,
            })

        return lx,z
    
np.random.seed(45)
z_init = np.load('samples/latent_vectors/032548.npy')[None, :]

G, D, Gs, sess = load_GAN_model(return_sess=True)
GI = GeneratorInverse(Gs, sess, learning_rate=.005, z_init=z_init)
GI.initialize()


print ("Starting training")

#f_image = 'samples/images/000360.jpg'
f_image = 'hoppe.jpg'
GI.set_target(f_image)


save_dest = 'samples/match_image'
os.system(f'rm -rvf {save_dest} && mkdir -p {save_dest}')

for i in range(200000):

    # Only save every 10 iterations
    if i%10==0:
        GI.render(f'{save_dest}/{i:05d}.jpg')

    loss, z = GI.train()
    norm = np.linalg.norm(z)/np.sqrt(512)
    print(loss, type(loss))
    print(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")
