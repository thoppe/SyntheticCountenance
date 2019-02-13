import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib

n_samples = 10 ** 5
random_seed = 42

f_z = 'samples/latent_vectors/00000042.npy'
z_base = np.load(f_z)

###################################################################

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()
save_dest_imgs = "samples/images_noise"
os.system(f"mkdir -p {save_dest_imgs}")

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))

# Observation: layer 4 is the hair wave!
alpha = 1.0

for k in tqdm(range(len(noise_vars))):
    for n in tqdm(range(20)):

        f_save = os.path.join(save_dest_imgs, f"level_{k:02d}_{n:08d}.jpg")

        noise_index = k
        nx = noise_vars[noise_index]

        small_noise = np.random.standard_normal(nx.shape)
        small_noise *= alpha
        tflib.set_vars({nx:small_noise})

        img, _, _ = generate_single(Gs, D, z=z_base)

        P_img = Image.fromarray(img)
        P_img.save(f_save)
