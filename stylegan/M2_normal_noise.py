import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib
import scipy.stats
import tensorflow as tf

'''
Need to get this to load and use the noise
'''

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

        

        img, *_ = generate_single(Gs, D, z=zx, compute_discriminator=False)

        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")
        scipy.misc.imsave(f_save, img)

##########################################################################

n_samples = 10 ** 5
random_seed = 42

name = 'donald_glover_0'
name = 'QuHarrison_whitebg'
f_dir = f'samples/match_image/{name}/*.npz'
f_npz = sorted(glob.glob(f_dir))[-1]

# Only even channels are interesting
# Channel 8 is interesting high resolution

frames_per = 30
epsilon = 0.60
n_transitions = 20

saved_vars = np.load(f_npz)
z0 = saved_vars['z']
noise = [saved_vars[f'noise_{n}'] for n in range(18)]

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

# Vicinity sampling around the center
delta = epsilon * np.random.standard_normal(size=(n_transitions - 1, 512))
    
Z = z0 + delta
save_dest = f"movie/noise_latents_{name}"
os.system(f"mkdir -p {save_dest}")
    
# Add the starting point back on for smooth loops 
Z = np.vstack([Z, Z[0]])

# Scale factor (see arXiv:1711.01970 Table 1)
Z /= np.sqrt(1 + epsilon ** 2)

sess = tf.get_default_session()
for key, val in zip(noise_vars, noise):
    print(f"Assigning noise {key}")
    sess.run(tf.assign(key,val))

interpolate(Z, frames_per=frames_per)
