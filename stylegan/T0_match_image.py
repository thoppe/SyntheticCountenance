"""
NEEDS WORK.


GAN inverter

Usage:
  T0_match_image.py <image_file> [--learning_rate=<f>] [--restart]

Options:
  -h --help     Show this screen.
  --learning_rate=<f>  ADAM learning rate [default: 0.0025]
  --restart
"""
from docopt import docopt

import dlib
import numpy as np
from PIL import Image
import os, json, glob
import dnnlib.tflib as tflib

import tensorflow as tf
from src.GAN_model import load_GAN_model, logger, generate_single, RGB_to_GAN_output

import cv2
import sklearn.decomposition
import PIL
from PIL import Image

np.random.seed(46)
n_image_upscale = 1


model_dest = "model/dlib"
shape68_pred = dlib.shape_predictor(
    f"{model_dest}/shape_predictor_68_face_landmarks.dat"
)
detector = dlib.get_frontal_face_detector()


def compute_keypoints(img, bbox):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape68 = shape68_pred(gray, bbox)
    keypoints = np.array(face_utils.shape_to_np(shape68))

    return keypoints


def compute_bbox(img, n_upsample=0):
    faces = detector(img, n_upsample)
    if len(faces) != 1:
        logger.warning(f"Found {len(faces)} faces in image! Expected one")
    return faces[0]


def single_img2latent_inference(f_image):

    clf = Image2Latent().load()
    raw_img = clf.preprocess_file(f_image)
    z = clf(raw_img)
    clf.sess.close()
    return z

def compute_convex_hull_face(img, eyes_only=False):
    """ Assume img is loaded from cv2.imread """

    bbox = compute_bbox(img, n_image_upscale)
    assert(len(bbox)==1)
    
    keypoints = KEY68(img, bbox[0])

    if eyes_only:
        keypoints = keypoints[36:48]

    hull = cv2.convexHull(keypoints)
    mask = np.zeros(img.shape, np.uint8)

    cv2.drawContours(mask, [hull], 0, 1, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask.astype(bool)[:, :, 0]

    return mask

    
    
def process_image(f_reference, f_image):
    # Align by the keypoints

    img0 = cv2.imread(f_reference)
    img1 = cv2.imread(f_image)

    img1 = scale_face(img0, img1)
    img1 = align_to_eyes(img0, img1)
    img1 = center_eyes(img1)

    # Center crop
    img1 = center_crop(img1)

    # Scale to the right size
    height, width = img1.shape[:2]
    if((height,width) != (1024,1014)):
        img1 = cv2.resize(img1, (1024, 1024))

    #bbox1 = compute_bbox(img1)[0]
    #k1 = KEY68(img1, bbox1).astype(np.float32)
    #for x,y in k1[36:48]:
    #    cv2.circle(img1,(x,y), 3, (255,255,255), -1)

    #for x,y in k_old[36:48]:
    #    cv2.circle(img1,(x,y), 3, (255,0,0), -1)
    
    # Blur around the face
    mask = compute_convex_hull_face(img1)
    img1[~mask] = cv2.blur(img1, (10,10))[~mask]
    img1[~mask] = cv2.blur(img1, (10,10))[~mask]
    
    # DEBUG here
    '''
    bbox0 = compute_bbox(img0)[0]
    k0 = KEY68(img0, bbox0).astype(np.float32)
    for x,y in k0[36:48]:
        cv2.circle(img1,(x,y), 3, (0,255,0), -1)

    cv2.imshow('img1', img1)
    cv2.waitKey(0)
    '''

    return img1
    


class GeneratorInverse:
    def __init__(self, generator, sess, learning_rate=0.01,
                 use_mask=True, z_init=None):

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
        
        #self.mask = tf.placeholder(dtype=tf.float32)

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
        #if use_mask:
        #    L1_loss *= self.mask

        self.loss = tf.reduce_sum(L1_loss ** 2)
        self.loss /= 1024**2
        #self.loss /= tf.reduce_sum(self.mask)

        #self.loss += tf.abs(1 - (tf.linalg.norm(self.z) / np.sqrt(latent_dim)))

        #self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.opt = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)

        # Only train the latent variable (hold the generator fixed!)
        self.train_op = self.opt.minimize(self.loss, var_list=[self.z])

    def initialize(self):
        self.sess.run(tf.initializers.variables([self.z]))
        self.sess.run(tf.initializers.variables(self.opt.variables()))

    def set_target(self, f_image):
        """
        For now, can only load from a file.
        """

        '''
        if self.use_mask:
            # Load a mask (all CV2 operations)
            cv_img = cv2.imread(f_image)
            mask = compute_convex_hull_face(cv_img)
            self.target_mask = np.ones_like(mask)

            # Set values off the mask to still be important
            self.target_mask[~mask] = 0.25

            # Set the values around the eyes to be twice as important
            eye_mask = compute_convex_hull_face(cv_img, eyes_only=True)
            self.target_mask[eye_mask] = 2.0
        else:
            # Debug line
            self.target_mask = np.ones((1024, 1024))
        '''
        
        # Load the target image
        img = Image.open(f_image)
        self.target_image = RGB_to_GAN_output(img)

    def render(self, f_save=None):
        """
        Renders the current latent vector into an image.
        """
        z_current = self.sess.run(
            self.z,
            feed_dict={
                self.img_in: self.target_image,
                #self.mask: self.target_mask
            },
        )
        
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        img = Gs.run(
            z_current,
            None,
            truncation_psi=0.7,
            use_noise=False,
            randomize_noise=False,
            output_transform=fmt
        )[0]

        print(img.shape)
        
        #img = Gs.run(z_current, np.zeros([512, 0]))
        #img = GAN_output_to_RGB(img)[0]

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
            feed_dict={
                self.img_in: self.target_image,
                #self.mask: self.target_mask
            },
        )


        return lx, z






if __name__ == "__main__":

    cargs = docopt(__doc__, version='GAN inverter 0.1')
    f_image = cargs['<image_file>']

    if not os.path.exists(f_image):
        logger.error(f"File not found, {f_image}")
        raise ValueError

    name = os.path.basename(f_image).split('.')[0]

    n_save_every = 10
    is_restart = False
    learning_rate = float(cargs['--learning_rate'])

    save_dest = f"samples/match_image/{name}"
    os.system(f'mkdir -p {save_dest}')

    f_processed = os.path.join(
        "samples/match_image/",
        f"match_{name}.jpg"
    )

    # Random init fool!
    #z_init = np.random.randn(512)

    G, D, Gs = load_GAN_model()
    sess = tf.get_default_session()
    
    GI = GeneratorInverse(Gs, sess, learning_rate=learning_rate, z_init=None)
    GI.initialize()

    logger.info(f"Starting training against {f_image}")
    GI.set_target(f_image)

    for i in range(0, 20000):

        # Only save every 10 iterations
        if i % n_save_every == 0:
            GI.render(f"{save_dest}/{i:05d}.jpg")

        loss, z = GI.train()
        norm = np.linalg.norm(z) / np.sqrt(512)
        logger.debug(f"Epoch {i}, loss {loss:0.4f}, z-norm {norm:0.4f}")
