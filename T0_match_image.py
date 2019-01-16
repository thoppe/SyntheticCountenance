import numpy as np
from PIL import Image
import os, json, glob
import tensorflow as tf
from src.GAN_model import load_GAN_model, logger
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output

from P3_keypoints_68 import compute_keypoints as KEY68
from P3_keypoints_5 import compute_keypoints as KEY5
from E0_closest_vector import latent_face_model

from P1_compute_bbox import compute_bbox

import cv2
import sklearn.decomposition

np.random.seed(45)
save_dest = "samples/match_image"
n_image_upscale = 1

def compute_convex_hull_face(img):
    """ Assume img is loaded from cv2.imread """

    bbox = compute_bbox(img, n_image_upscale)
    assert(len(bbox)==1)
    
    keypoints = KEY68(img, bbox[0])
    hull = cv2.convexHull(keypoints)
    mask = np.zeros(img.shape, np.uint8)

    cv2.drawContours(mask, [hull], 0, 1, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(bool)[:, :, 0]

    return mask

def center_eyes(img):
    
    # From the keypoints around the eyes, make them on the x-axis
    bbox = compute_bbox(img, n_image_upscale)
    k1 = KEY68(img, bbox[0]).astype(np.float32)

    vx = sklearn.decomposition.PCA(
        n_components=1).fit(k1[36:48]).components_[0]

    # SVD does not determine the direction, but we have one
    if vx[0] < 0:
        vx *= -1
    #for x,y in k1[36:48]:
    #    cv2.circle(img,(x,y), 3, (255,255,255), -1)
    #cv2.imshow('img', img)
    #cv2.waitKey(0)

    theta = np.arctan2(vx[1], vx[0]) * (180.0/np.pi)
    logger.info(f"Rotating input image by {theta:0.2f} degrees")

    center = tuple(np.array([1024,1024])/2)
    rot_mat = cv2.getRotationMatrix2D(center, theta, 1.0)
    img_out = cv2.warpAffine(
        img, rot_mat, (1024,1024),
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return img_out

def align_to_eyes(img0, img1):
    # Using the eye keypoints, find the optimal shift from img0 -> img1
    
    bbox0 = compute_bbox(img0, n_image_upscale)[0]
    k0 = KEY68(img0, bbox0).astype(np.float32)

    bbox1 = compute_bbox(img1, n_image_upscale)[0]
    k1 = KEY68(img1, bbox1).astype(np.float32)
    
    # Get the average eye keypoints
    left_eye0 = k0[36:42].mean(axis=0)
    left_eye1 = k1[36:42].mean(axis=0)

    right_eye0 = k0[42:48].mean(axis=0)
    right_eye1 = k1[42:48].mean(axis=0)
    
    shift = np.array([
        -(left_eye1 - left_eye0),
        -(right_eye1 - right_eye0),
    ]).mean(axis=0)

    M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
    logger.info(f"Shifting input image by {shift} pixels")
    
    img_out = cv2.warpAffine(
        img1,M,(1024, 1024),
        borderMode=cv2.BORDER_REFLECT_101
    )

    return img_out
    
    
def process_image(f_reference, f_image):
    # Align by the keypoints

    img0 = cv2.imread(f_reference)
    img1 = cv2.imread(f_image)

    height, width = img1.shape[:2]

    # Center crop
    if(height != width):
        logger.warning("Center cropping image")
        idim = min(height, width)
        delta = abs(height - width)
        
        if height > width:
            if delta%2==0:
                img1 = img1[delta//2:-delta//2,:]
            else:
                img1 = img1[delta//2:delta//2+1,:]
        else:
            if delta%2==0:
                img1 = img1[:,delta//2:-delta//2]
            else:
                img1 = img1[:,delta//2:delta//2+1]

        height, width = img1.shape[:2]
        assert(height==width)

    # Scale to the right size
    if((height,width) != (1024,1014)):
        img1 = cv2.resize(img1, (1024, 1024))

    img1 = center_eyes(img1)
    img1 = align_to_eyes(img0, img1)

    #bbox1 = compute_bbox(img1)[0]
    #k1 = KEY68(img1, bbox1).astype(np.float32)
    #for x,y in k1[36:48]:
    #    cv2.circle(img1,(x,y), 3, (255,255,255), -1)

    #for x,y in k_old[36:48]:
    #    cv2.circle(img1,(x,y), 3, (255,0,0), -1)
    #bbox0 = compute_bbox(img0)[0]
    #k0 = KEY68(img0, bbox0).astype(np.float32)
    #for x,y in k0[36:48]:
    #    cv2.circle(img1,(x,y), 3, (0,255,0), -1)
    
    # Blur around the face
    mask = compute_convex_hull_face(img1)
    img1[~mask] = cv2.blur(img1, (10,10))[~mask]
    img1[~mask] = cv2.blur(img1, (10,10))[~mask]

    #cv2.imshow('img0', img0)
    #cv2.imshow('img1', img1)
    #cv2.waitKey(0)
    #exit()

    return img1
    


class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01, use_mask=True, z_init=None):

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
        label_dummy = tf.zeros([batch_size, 0])

        G_out = generator.get_output_for(self.z, label_dummy, is_training=False)

        # NCHW
        self.img_in = tf.placeholder(
            tf.float32, shape=(batch_size, 3, image_dim, image_dim)
        )

        # L1 error, only train the loss
        L1_loss = tf.abs(G_out - self.img_in)

        # Sum over the batch_size, channel info
        L1_loss = tf.reduce_sum(L1_loss, axis=0)
        L1_loss = tf.reduce_sum(L1_loss, axis=0)

        # Sum over all pixels
        if use_mask:
            L1_loss *= self.mask

        self.loss = tf.reduce_sum(L1_loss ** 2)
        self.loss /= tf.reduce_sum(self.mask)

        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # self.opt = tf.train.GradientDescentOptimizer(
        # learning_rate=learning_rate)

        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.z])

    def initialize(self):
        self.sess.run(tf.initializers.variables([self.z]))

        self.sess.run(tf.initializers.variables(self.opt.variables()))

    def set_target(self, f_image):
        """
        For now, can only load from a file.
        """

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
        """
        Renders the current latent vector into an image.
        """
        z_current = self.sess.run(
            self.z,
            feed_dict={self.img_in: self.target_image, self.mask: self.target_mask},
        )

        img = Gs.run(z_current, np.zeros([512, 0]))
        img = GAN_output_to_RGB(img)[0]

        if f_save is not None:
            P_img = Image.fromarray(img)
            P_img.save(f_save)

            f_npy = f_save.replace(".jpg", ".npy")
            np.save(f_npy, z_current)

        return img

    def train(self):
        """
        For each training step make sure the raw image has been
        loaded with RGB_to_GAN_output. Returns the loss.
        """

        if self.target_image is None:
            raise ValueError("must call .set_target(img) first!")

        outputs = [self.loss, self.z, self.train_op]
        lx, z, _ = self.sess.run(
            outputs,
            feed_dict={self.img_in: self.target_image, self.mask: self.target_mask},
        )

        return lx, z





#f_image = '/home/travis/Desktop/Dw1uV7vWsAEuKwv.jpg'
#f_image = '/home/travis/Desktop/hoppe.jpg'
#f_image = '/home/travis/Desktop/1_LYJ80Dx2rTnPvg7Kk0u4sA.jpg'
#f_image = '/home/travis/Desktop/ChristinaF.jpg'
f_image = '/home/travis/Desktop/rihanna.jpg'

latent_starting_offset = 0

is_restart = False
learning_rate = 0.01

if not is_restart:
    # Find the closest matching image as our starting conditions
    LFM = latent_face_model()
    n_reference = LFM.closest_index(f_image)[latent_starting_offset]
    logger.info(
        f"Image index {n_reference} has the closest matching facial features")

    z_init = np.load(f"samples/latent_vectors/{n_reference:06d}.npy")[None, :]
    f_reference = f'samples/images/{n_reference:06d}.jpg'
    start_idx = 0
    os.system(f"rm -rf {save_dest} && mkdir -p {save_dest}")
    
else:
    import glob
    f_z = sorted(glob.glob("samples/match_image/*.npy"))[-1]
    z_init = np.load(f_z)
    learning_rate /= 2
    start_idx = int(os.path.basename(f_z).split('.')[0])


# Preprocess the image
f_processed = f"samples/match_target.jpg"

if not is_restart:
    img_process = process_image(f_reference, f_image)
    cv2.imwrite(f_processed, img_process)




G, D, Gs, sess = load_GAN_model(return_sess=True)
GI = GeneratorInverse(Gs, sess, learning_rate=learning_rate, z_init=z_init)
GI.initialize()

logger.info(f"Starting training against {f_processed}")
GI.set_target(f_processed)


for i in range(start_idx, 20000):

    # Only save every 10 iterations
    if i % 10 == 0:
        GI.render(f"{save_dest}/{i:05d}.jpg")

    loss, z = GI.train()
    norm = np.linalg.norm(z) / np.sqrt(512)
    logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")
