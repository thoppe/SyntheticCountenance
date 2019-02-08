import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import scipy.stats

def interpolate(Z, W):
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

        wt = (W.T * weights).sum(axis=2)
        wt /= np.sqrt((weights ** 2).sum())
        wt = wt[np.newaxis, np.newaxis, :, :]

        wt *= frame_idx/100.0

        zt = (Z.T * weights).sum(axis=1)
        zt /= np.sqrt((weights ** 2).sum())

        tflib.set_vars({noise_vars[noise_channel]:wt})
        img, _, _ = generate_single(Gs, None, z=zt)

        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")
        scipy.misc.imsave(f_save, img)

##########################################################################

n_samples = 15
frames_per = 15
random_seed = 40

alpha = 225.0 * 2
noise_channel = 2

# Solid base alpha is 450

z_dim = 512

G, D, Gs = load_GAN_model()

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

#for noise_channel in range(len(noise_vars)):
#for noise_channel in [2]:
for random_seed in range(100, 200):
#for alpha in range(0, 500, 5):
    np.random.seed(random_seed)

    #save_dest = f"movie/high_noise_{random_seed:04d}"
    save_dest = f"movie/alpha_noise_{random_seed:04d}"
    
    os.system(f"mkdir -p {save_dest}")

    # Reshape to drop first two channels
    noise_shape = noise_vars[noise_channel].shape[2:]
    print("Noise shape", noise_shape)

    # The starting latent vectors
    Z = [np.random.standard_normal(z_dim) for _ in range(n_samples-1)]

    # The noise vectors
    # Boost the f* out of a single vector
    W = [alpha*np.random.standard_normal(noise_shape) for _ in range(len(Z))]

    # Fix w? Fix z?
    W = [W[0] for _ in range(len(Z))]
    Z = [Z[0] for _ in range(len(Z))]

    # Add the starting point back on for smooth loops 
    Z.append(Z[0])
    W.append(W[0])

    Z = np.array(Z)
    W = np.array(W)

    interpolate(Z, W)
