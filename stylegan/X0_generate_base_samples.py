import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import scipy.stats
import tensorflow as tf


def interpolate(Z, frames_per=30):
    n = len(Z)
    T = np.linspace(0, n - 1, n * frames_per)

    low_s = 0.75
    high_s = 1.00

    sigma = np.random.uniform(low=low_s, high=high_s, size=[n,])
    
    G = [scipy.stats.norm(loc=k, scale=s) for k,s in zip(range(n), sigma)]

    for frame_idx, t in tqdm(enumerate(T), total=len(T)):
        weights = np.array([g.pdf(t) for g in G])

        # Need to wrap around weights for perfect loop
        weights += np.array([g.pdf(t + T.max()) for g in G])
        weights += np.array([g.pdf(t - T.max()) for g in G])

        # Set the strength of the weights to be unity
        weights /= weights.sum()

        zx = (Z.T * weights).sum(axis=1)
        zx /= np.sqrt((weights ** 2).sum())

        img, *_ = generate_single(Gs, None, z=zx, compute_discriminator=False)

        yield img


##########################################################################

n_transitions = 6
n_people = 100
frames_per = 15
epsilon = 0.80
random_seed_offset = 2342
noise_channel = 2
alpha= 325.0

f_dir0 = f'monstergan/base_images'
f_dir1 = f'monstergan/noisy_images'

os.system(f"mkdir -p {f_dir0}")
os.system(f"mkdir -p {f_dir1}")


G, D, Gs = load_GAN_model()

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]
noise_shape = noise_vars[noise_channel].shape

for person_idx in tqdm(range(n_people)):
    np.random.seed(random_seed_offset + person_idx)

    f_check = os.path.join(f_dir0, f"{person_idx:04d}_{0:06d}.jpg")
    if os.path.exists(f_check):
        print(f"Skipping {person_idx:04d}")
        continue

    # Single noise per person
    noise = np.random.standard_normal(noise_shape)

    z0 = np.random.standard_normal(size=(512,))

    # Vicinity sampling around the center
    delta = epsilon * np.random.standard_normal(size=(n_transitions - 1, 512))
    
    Z = z0 + delta
    print(Z.shape)
    
    # Add the starting point back on for smooth loops 
    Z = np.vstack([Z, Z[0]])

    # Scale factor (see arXiv:1711.01970 Table 1)
    Z /= np.sqrt(1 + epsilon ** 2)

    ITR = interpolate(Z, frames_per=frames_per)
    tflib.set_vars({noise_vars[noise_channel]:noise})
    for frame_idx, img in tqdm(enumerate(ITR)):
        f_save = os.path.join(f_dir0, f"{person_idx:04d}_{frame_idx:06d}.jpg")
        scipy.misc.imsave(f_save, img)

    ITR = interpolate(Z, frames_per=frames_per)
    tflib.set_vars({noise_vars[noise_channel]:alpha*noise})
    for frame_idx, img in tqdm(enumerate(ITR)):
        f_save = os.path.join(f_dir1, f"{person_idx:04d}_{frame_idx:06d}.jpg")
        scipy.misc.imsave(f_save, img)

