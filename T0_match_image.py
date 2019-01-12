import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output


class GeneratorInverse:

    def __init__(self, generator, sess, learning_rate=0.01):

        self.sess = sess

        latent_dim = 512
        image_dim = 1024
        batch_size = 1

        # Start with random init for the latents
        z_init = np.random.randn(latent_dim)
        self.z = tf.Variable(z_init[None, :], dtype=tf.float32)

        # Labels are not needed for this project
        label_dummy = tf.zeros([batch_size,0])
    
        G_out = generator.get_output_for(
            self.z, label_dummy, is_training=False)

        # NCHW
        self.img_in = tf.placeholder(
            tf.float32, shape=(batch_size, 3, image_dim, image_dim))

        # L1 error, only train the loss
        self.loss = tf.reduce_sum(tf.abs(G_out - self.img_in))
        self.loss /= image_dim**2
    
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.z,])

    def initialize(self):
        self.sess.run(
            tf.initializers.variables([self.z,]))
        
        self.sess.run(
            tf.initializers.variables(self.opt.variables()))
        
    def train(self, input_image_grid):
        '''
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        '''
        
        outputs = [self.loss, self.train_op]
        lx,_ = self.sess.run(
            outputs, feed_dict={self.img_in:input_image_grid})

        return lx
    

G, D, Gs, sess = load_GAN_model(return_sess=True)

GI = GeneratorInverse(Gs, sess)
GI.initialize()
            


# Load in the target image and bring it proper format
# Open the image and scale it to [-1, 1] and CHW
f_image = 'src/000260.jpg'
img = Image.open(f_image)
img_grid = RGB_to_GAN_output(img)

#f_npy = 'samples/latent_vectors/000360.npy'
#z0 = np.load(f_npy)

print ("Starting training")

save_dest = 'training_demo'

for i in range(200000):

    if i%10==0:

        z_current = sess.run(GI.z, feed_dict={GI.img_in:img_grid})
        img = Gs.run(z_current, np.zeros([512,0]))

        img = GAN_output_to_RGB(img)[0]
        P_img = Image.fromarray(img)
        f_save = f'{save_dest}/{i:05d}.jpg'

        if i == 0:
            os.system(f'rm -vf {save_dest}/*')
        
        P_img.save(f_save)

    loss = GI.train(img_grid)
    print(i, loss)
