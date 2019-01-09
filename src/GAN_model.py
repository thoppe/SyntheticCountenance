import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json


def load_GAN_model():
    print("Loading the model")

    import tensorflow as tf
    import sys
    import pickle

    dim = 512

    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45
    sess = tf.InteractiveSession(config=config)

    f_model = 'model/karras2018iclr-celebahq-1024x1024.pkl'
    path_pg_gan_code = 'src/model/pggan/'

    sys.path.append(path_pg_gan_code)
    with open(f_model, 'rb') as FIN:
        G, D, Gs = pickle.load(FIN)

    return G, D, Gs

def generate_single(
        generator, discriminator, z=None, compute_discriminator=True):
    '''
    Pass in a generator, discriminator and optionally a latent vector

    Returns (image, latent_vector, discriminator_score,)
    
    '''
    if z is None:
        z = np.random.randn(dim)

    zf = z[None, :]
    dummy = np.zeros([z.shape[0], 0])

    img = generator.run(zf, dummy)

    if compute_discriminator:
        ds = discriminator.run(img)[0].ravel()[0]
    else:
        ds = None

    # [-1,1] => [0,255]
    img = np.clip(np.rint(
        (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC

    return img[0], z, ds