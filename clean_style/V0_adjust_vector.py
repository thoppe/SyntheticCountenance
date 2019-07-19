'''
Must run B4 first.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pixelhouse as ph
import os, json, glob

from src.GAN_model import load_GAN_model, generate_single
G, D, Gs = load_GAN_model()


target_idx = 54

random_seed = 42
np.random.seed(random_seed)

f_latent = f'data/latent_vectors/{target_idx:08d}.npy'
z = np.load(f_latent)
z_norm = np.linalg.norm(z)

FEATURES = [
    os.path.basename(f).split('.')[0] for f in glob.glob('averages/*.json')]

cv = ph.Canvas()

for feature in FEATURES:
    
    print(f"Starting {feature}")

    f_dz = f'averages/{feature}.json'
    with open(f_dz) as FIN:
        js = json.load(FIN)
        dz = np.array(js['latent'])

    dz /= np.linalg.norm(dz)
    dz *= z_norm

    save_dest = f"data/adjusted_images/{feature}"
    os.system(f"mkdir -p {save_dest}")

    for delta in np.linspace(-0.30, 0.30, 50):

        zx = (z + delta*dz) / z_norm
        delta += 1

        f_save = os.path.join(save_dest, f"{target_idx:08d}_{delta:0.4f}.jpg")

        img, _, noise = generate_single(Gs, D, z=zx,
                                        compute_discriminator=False)

        cv.img = img
        cv += ph.text(x=2, y=-3.0, text=f"{delta:0.3f}\n{feature}",
                      font_size=0.25)

        
        cv.save(f_save)       
        cv.resize(0.5).show(1)

