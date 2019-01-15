import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json
import coloredlogs
import logging
coloredlogs.install(level='DEBUG')

dim = 512

# Create a logger object.
logger = logging.getLogger(__name__)

def load_GAN_model(return_sess=False):
    logger.info("Loading the PGAN model")

    import tensorflow as tf
    import sys
    import pickle

    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.InteractiveSession(config=config)
    
    if not tf.test.is_gpu_available():
        logger.error(
            f"The PGAN model requires a GPU (some instructions do not work on CPU)")
        raise NotImplementedError

    f_model = "model/karras2018iclr-celebahq-1024x1024.pkl"
    path_pg_gan_code = "src/model/pggan/"

    sys.path.append(path_pg_gan_code)

    if not os.path.exists(f_model):
        logger.error(f"Can't find model file {f_model}, use README to find download link.")
        raise(FileNotFoundError)
    
    with open(f_model, "rb") as FIN:
        G, D, Gs = pickle.load(FIN)

    if not return_sess:
        return G, D, Gs

    return G, D, Gs, sess


def generate_single(generator, discriminator, z=None, compute_discriminator=True):
    """
    Pass in a generator, discriminator and optionally a latent vector

    Returns (image, latent_vector, discriminator_score,)
    
    """
    if z is None:
        z = np.random.randn(dim)

    zf = z[None, :]
    dummy = np.zeros([z.shape[0], 0])

    img = generator.run(zf, dummy)

    if compute_discriminator:
        ds = discriminator.run(img)[0].ravel()[0]
    else:
        ds = None

    img = GAN_output_to_RGB(img)[0]

    return img, z, ds


def GAN_output_to_RGB(img):

    # [-1,1] => [0,255]
    img = np.clip(np.rint((img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC
    return img


def RGB_to_GAN_output(img):
    batch_size = 1

    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = 2 * (img / 255.0) - 1
    grid = np.tile(img, (batch_size, 1, 1, 1))

    return grid
