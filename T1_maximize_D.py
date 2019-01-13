'''
Note to self: This is a really good idea! I just don't have the GPU memory to
do it...

The idea is to find the z such that D(G(z)) is minimized, that is the picture
that _most_ fools the discriminator.
'''

import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output


run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

class MaximizeDiscriminator:

    def __init__(self, generator, discriminator, sess, learning_rate=0.01):

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

        D_scores, D_labels = fp32(discriminator.get_output_for(
            G_out, is_training=False))

        #self.loss = tf.reduce_sum(D_scores)
        self.loss = D_scores
        print(self.loss)
        
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.z,])

    def initialize(self):
        self.sess.run(
            tf.initializers.variables([self.z,]))
        
        self.sess.run(
            tf.initializers.variables(self.opt.variables()))

        tf.graph_util.remove_training_nodes(
            tf.get_default_graph().as_graph_def(),
        )

    def render(self, f_save=None):
        '''
        Renders the current latent vector into an image.
        '''
        z_current = self.sess.run(self.z)

        img = Gs.run(z_current, np.zeros([512,0]))
        img = GAN_output_to_RGB(img)[0]

        if f_save is not None:
            P_img = Image.fromarray(img)        
            P_img.save(f_save)

        return img
        
    def train(self):
        '''
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        '''
        
        outputs = [self.loss, self.z, self.train_op]
        lx,z,_ = self.sess.run(outputs)

        return lx,z
    
np.random.seed(45)
G, D, Gs, sess = load_GAN_model(return_sess=True)
MD = MaximizeDiscriminator(Gs, D, sess, learning_rate=0.001)
MD.initialize()

print ("Starting training")
save_dest = 'samples/maximizer_demo'

os.system(f'rm -rvf {save_dest} && mkdir -p {save_dest}')

for i in range(200000):

    # Only save every 10 iterations
    #if i%10==0:
    #    MD.render(f'{save_dest}/{i:05d}.jpg')

    loss, z = MD.train()
    norm = np.linalg.norm(z)/np.sqrt(512)
    print(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")

    
