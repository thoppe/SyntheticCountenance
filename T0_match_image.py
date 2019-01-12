import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model, raw_GAN_output_to_RGB
import h5py

latent_dim = 512
image_dim = 1024
batch_size = 1

G, D, Gs, sess = load_GAN_model(return_sess=True)

#z0 = np.random.randn(latent_dim)
f_npy = 'samples/latent_vectors/000360.npy'
z0 = np.load(f_npy)

label_dummy = tf.zeros([batch_size,0])
latents = tf.Variable(z0[None, :], dtype=tf.float32)
G_out = Gs.get_output_for(latents, label_dummy, is_training=False)

# NCHW
target_images = tf.placeholder(
    tf.float32, shape=(batch_size, 3, image_dim, image_dim))

# L1 error, only train the loss
loss = tf.reduce_sum(tf.abs(G_out - target_images))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, var_list=[latents,])

# Load in the target image and bring it proper format
f_image = 'src/000260.jpg'

# Open the image and scale it to [-1, 1] and CHW
img = Image.open(f_image)
img = np.array(img).transpose(2,0,1).astype(float)
img = 2*(img/255.0) - 1
grid = np.tile(img, (batch_size, 1, 1, 1))

print ("Starting training")

init_new_vars_op = tf.initializers.variables([latents,])
sess.run(init_new_vars_op)

init_new_vars_op = tf.initializers.variables(optimizer.variables())
sess.run(init_new_vars_op)

lx = sess.run(loss, feed_dict={target_images:grid})
print(f"Starting loss: {lx / image_dim**2}")

save_dest = 'training_demo'

for i in range(200000):

    if i%10==0:

        z_current = sess.run(latents, feed_dict={target_images:grid})

        print("Z_ERROR", np.linalg.norm(z0-z_current))
        img = Gs.run(z_current, np.zeros([512,0]))

        img = raw_GAN_output_to_RGB(img)[0]
        P_img = Image.fromarray(img)
        f_save = f'{save_dest}/{i:05d}.jpg'

        if i == 0:
            os.system(f'rm -vf {save_dest}/*')
        
        P_img.save(f_save)
        #print(f_save)
        #exit()
        
    outputs = [loss, train_op]
    lx,_ = sess.run(outputs, feed_dict={target_images:grid})
    lx /= image_dim**2

    
    print(i, lx)
    

#with tf.Session() as sess:
#    writer = tf.summary.FileWriter('logs', sess.graph)
#    #print (sess.run(Ggolden_ratio))
#    writer.close()

