"""
The idea is to find the img such that D(img) is minimized, that is the picture
that _most_ fools the discriminator.
"""

import numpy as np
import os, json, glob, random, h5py
from tqdm import tqdm
import tensorflow as tf
from src.GAN_model import GAN_output_to_RGB, RGB_to_GAN_output


from src.img2latent import Image2Latent

def image_pipeline(batch_size=5):

    f_h5 = 'samples/PGAN_small_images.h5'
    with h5py.File(f_h5, 'r') as h5:
        N = len(h5['Z'])
        Z = h5['Z'][...]

        while True:
            idx = np.random.randint(0, N, size=batch_size)
            img = np.array([h5['IMG'][i] for i in idx])

            img = RGB_to_GAN_output(img, batch_size=batch_size, resize=False)
            yield Z[idx], img



if __name__ == "__main__":
    batch_size = 32
    n_epochs = 2000
    n_save_every = 50

    ITR = image_pipeline(batch_size)
    clf = Image2Latent(batch_size=batch_size)

    while True:

        for n, (z,img) in enumerate(ITR):

            if n%n_save_every == 0:
                clf.render(z, img)
                #s = clf.save()

            lx = clf.train(z, img)
            print(n, lx)
