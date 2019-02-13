import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib

n_samples = 20000
random_seed = 29

###################################################################

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()
save_dest_imgs = "samples/images_noise"
os.system(f"mkdir -p {save_dest_imgs}")

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

noise_index = 2

# Observation: layer 4 is the hair wave!
alpha = 320.0

for n in tqdm(range(n_samples)):

    z_base = np.random.standard_normal(512)
    nx = noise_vars[noise_index]

    noise = np.random.standard_normal(nx.shape)

    tflib.set_vars({nx:noise})
    img, _, _ = generate_single(Gs, D, z=z_base)

    f_save = os.path.join(save_dest_imgs, f"level_{n:08d}_a.jpg")
    Image.fromarray(img).save(f_save)


    tflib.set_vars({nx:alpha*noise})
    img, _, _ = generate_single(Gs, D, z=z_base)

    f_save = os.path.join(save_dest_imgs, f"level_{n:08d}_b.jpg")
    Image.fromarray(img).save(f_save)
    
