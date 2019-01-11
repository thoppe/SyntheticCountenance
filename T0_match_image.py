import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model, generate_single
import h5py

latent_dim = 512
image_dim = 1024
batch_size = 1

G, D, Gs, sess = load_GAN_model(return_sess=True)
dummy = tf.zeros([batch_size,0])

#z0 = np.random.randn(latent_dim)
f_npy = 'samples/latent_vectors/000360.npy'
z0 = np.load(f_npy)



latents = tf.Variable(z0[None, :], dtype=tf.float32)
# Keep the latents near the expected value
#latents /= tf.norm(latents)
#latents *= np.sqrt(latent_dim)

# For whatever reason, I'm getting different values from Gs.get_output_for
# and Gs.run(). This is the training difference ...
G_out = Gs.get_output_for(latents, dummy, is_training=False)
sess.run(tf.global_variables_initializer())
img = sess.run(G_out)

###########################################################
#dummy = np.zeros([512,0])
#img = Gs.run(z0[None, :], dummy)

img = np.clip(np.rint(
    (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC
P_img = Image.fromarray(img[0])
P_img.show()
exit()

###########################################################



# NCHW
target_images = tf.placeholder(
    tf.float32, shape=(batch_size, 3, image_dim, image_dim))

# L1 error
loss = tf.reduce_sum(tf.abs(G_out - target_images))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Load in the target image and bring it proper format
f_image = 'src/000260.jpg'

# Open the image and scale it to [-1, 1] and CHW
img = Image.open(f_image)
img = np.array(img).transpose(2,0,1).astype(float)
img = 2*(img/255.0) - 1
grid = np.tile(img, (batch_size, 1, 1, 1))

sess.run(tf.global_variables_initializer())

print (G_out)
print (target_images)
print (loss)
print ("Starting training")

lx,img_out = sess.run(
    [loss, G_out], feed_dict={target_images:grid})

print(f"Starting loss: {lx / image_dim**2}")
print(img_out)

save_dest = 'training_demo'

for i in range(200000):

    if i%10==0:
        #img = sess.run(G_out, feed_dict={target_images:grid})
        img = sess.run(Gs, z0)
        
        img = np.clip(np.rint(
            (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
        img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC
        P_img = Image.fromarray(img[0])
        f_save = f'{save_dest}/{i:05d}.jpg'

        if i == 0:
            os.system(f'rm {save_dest}/*')
        
        P_img.save(f_save)
        print(f_save)
        exit()
        
    outputs = [loss, train_op]
    lx,_ = sess.run(outputs, feed_dict={target_images:grid})
    lx /= image_dim**2

    
    print(i, lx)
    


