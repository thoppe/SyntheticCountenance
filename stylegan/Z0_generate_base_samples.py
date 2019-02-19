import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import scipy.stats
import tensorflow as tf


def interpolate(Z, NOISE, frames_per=30):
    
    n = len(NOISE[0])
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

        img, *_ = generate_single(Gs, None, z=zx)

        yield img


##########################################################################

n_transitions = 10
n_people = 100
frames_per = 20
epsilon = 00.50

f_dir0 = f'tripgan/base_images'
os.system(f"mkdir -p {f_dir0}")

z_base_index = 27
f_z = f'samples/latent_vectors/{z_base_index:08d}.npy'
assert(os.path.exists(f_z))
z0 = np.load(f_z)

G, D, Gs = load_GAN_model()

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

noise_sizes = [nx.shape for nx in noise_vars]

        #for nx, tf_var in zip(NOISE, noise_vars):
        #    nx = (nx.T*weights).sum(axis=1)
        #    nx /= np.sqrt((weights ** 2).sum())
        #    nx = nx.reshape(tf_var.shape)
        #    tflib.set_vars({tf_var: nx})




for person_idx in tqdm(range(n_people)):
    np.random.seed(person_idx)

    f_check = os.path.join(f_dir0, f"{person_idx:04d}_{179:06d}.jpg")
    if os.path.exists(f_check):
        print(f"Skipping {person_idx:04d}")
        continue

    # Set the base noise for each transistion
    NOISE = [ ]
    for nx in noise_vars:
        noise_length = np.prod(nx.shape)
        print("Setting noise", noise_length)

        NC = epsilon*np.random.standard_normal(
               size=(n_transitions - 1, noise_length))

        # Add the starting point back on for smooth loops
        NC = np.vstack([NC, NC[0]])

        # Scale factor (see arXiv:1711.01970 Table 1)
        NC /= np.sqrt(1 + epsilon ** 2)
        
        NOISE.append(NC)


    delta = epsilon * np.random.standard_normal(size=(n_transitions - 1, 512))
    Z = z0 + delta     
    # Add the starting point back on for smooth loops 
    Z = np.vstack([Z, Z[0]])

    # Scale factor (see arXiv:1711.01970 Table 1)
    Z /= np.sqrt(1 + epsilon ** 2)
    

    ITR = interpolate(Z, NOISE, frames_per=frames_per)

    for frame_idx, img in tqdm(enumerate(ITR)):
        f_save = os.path.join(f_dir0, f"{person_idx:04d}_{frame_idx:06d}.jpg")
        scipy.misc.imsave(f_save, img)
