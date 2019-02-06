import os, json, pickle
from tqdm import tqdm
from PIL import Image
import numpy as np
import dnnlib.tflib as tflib

import coloredlogs
import logging
coloredlogs.install(level='DEBUG')

dim = 512

# Create a logger object.
logger = logging.getLogger(__name__)

#result_dir = 'results'
#data_dir = 'datasets
#run_dir_ignore = ['results', 'datasets', 'cache']
cache_dir = 'model'


def load_GAN_model(return_sess=False, sess=None):
    logger.info("Loading the ffhq stylegan model")

    import dnnlib

    tflib.init_tf()

    # Load pre-trained network.
    # karras2019stylegan-ffhq-1024x1024.pkl
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    
    with dnnlib.util.open_url(url, cache_dir=cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    return _G, _D, Gs


def generate_single(
        generator, discriminator, z=None, compute_discriminator=False):
    """
    Pass in a generator, discriminator and optionally a latent vector

    Returns (image, latent_vector, discriminator_score,)
    
    """
    if z is None:
        z = np.random.randn(dim)

    zf = z[None, :]
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    img = generator.run(
        zf,
        None,
        truncation_psi=0.7,
        randomize_noise=False,
        output_transform=fmt
    )

    # Take a single image
    img = img[0]
    
    if compute_discriminator:
        ds = discriminator.run(img)[0].ravel()[0]
    else:
        ds = None

    return img, z, ds
