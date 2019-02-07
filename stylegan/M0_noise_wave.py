import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import scipy.stats

def interpolate(W):
    n = len(W)
    T = np.linspace(0, n - 1, n * frames_per)

    low_s = 0.50
    high_s = 0.75
   
    sigma = np.random.uniform(low=low_s, high=high_s, size=[n,])
    
    G = [scipy.stats.norm(loc=k, scale=s) for k,s in zip(range(n), sigma)]

    for frame_idx, t in tqdm(enumerate(T), total=len(T)):
        weights = np.array([g.pdf(t) for g in G])

        # Need to wrap around weights for perfect loop
        weights += np.array([g.pdf(t + T.max()) for g in G])
        weights += np.array([g.pdf(t - T.max()) for g in G])

        # Set the strength of the weights to be unity
        weights /= weights.sum()

        wx = (W.T * weights).sum(axis=2)
        wx /= np.sqrt((weights ** 2).sum())

        noise = wx[np.newaxis, np.newaxis, :, :]
        tflib.set_vars({noise_vars[noise_channel]:noise})
                
        img, _, _ = generate_single(Gs, None, z=z_base)

        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")
        scipy.misc.imsave(f_save, img)

##########################################################################

n_samples = 10 ** 5
random_seed = 42

nz = 42

# 374 older man outside, good contrast

# Only even channels are interesting
# Channel 8 is interesting high resolution

noise_channel = 5
noise_samples = 60
frames_per = 10
alpha = 4.0

f_z = f'samples/latent_vectors/{nz:08d}.npy'
z_base = np.load(f_z)

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

for noise_channel in range(len(noise_vars)):
    print(noise_channel)

    save_dest = f"movie/noise_{nz:08d}_{noise_channel:02d}"
    os.system(f"mkdir -p {save_dest}")

    noise_shape = noise_vars[noise_channel].shape
    print("Known noise shape", noise_shape)

    # Reshape to drop first two channels
    noise_shape = noise_shape[2:]

    W = [
        (alpha*(k+1)) *
        np.random.standard_normal(noise_shape) for k in range(noise_samples-1)]

    # Add in a normal bit near the end cause it's too crazy
    W.append(np.random.standard_normal(noise_shape))

    # Add the starting point back on for smooth loops 
    W.append(W[0])

    W = np.array(W)

    interpolate(W)
