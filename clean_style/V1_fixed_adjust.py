'''
Must run B4 first.
'''

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pixelhouse as ph
import os, json, glob
from scipy.spatial.distance import cosine

from src.GAN_model import load_GAN_model, generate_single
G, D, Gs = load_GAN_model()


target_idx = 31

random_seed = 42
np.random.seed(random_seed)

f_latent = f'data/latent_vectors/{target_idx:08d}.npy'
z = np.load(f_latent)
z_norm = np.linalg.norm(z)
cv = ph.Canvas()


F_FEATURES = [
    os.path.basename(f).split('.')[0] for f in glob.glob('averages/*.json')
]

F_FEATURES = [f for f in F_FEATURES if f in ["age", "female_score"]]


FEATURES = {}
for feature in F_FEATURES:
    f_dz = f'averages/{feature}.json'
    
    with open(f_dz) as FIN:
        js = json.load(FIN)
        dz = np.array(js['latent'])
        dz *= (z_norm/np.linalg.norm(dz))

    FEATURES[feature] = dz

for name in FEATURES:

    dz = FEATURES[name]
    norm = np.linalg.norm(dz)

    delta_vecs = []
    for other in FEATURES:

        if other not in ['age', 'female_score']:
            continue
        
        if name == other:
            continue
        oz = FEATURES[other]
        dist = 1-cosine(dz, oz)

        if dist < 0:
            continue

        print(name, other, dist)

        delta_vecs.append(oz*dist)

    for vec in delta_vecs:
        dz -= vec
        
    dz /= norm


for feature in F_FEATURES:
    
    print(f"Starting {feature}")
    dz = FEATURES[feature]
    
    save_dest = f"data/adjusted_fixed/{feature}"
    os.system(f"mkdir -p {save_dest}")

    for delta in np.linspace(-3, 3, 50):

        zx = z + delta*dz
        #zx /= z_norm

        f_save = os.path.join(save_dest, f"{target_idx:08d}_{delta:0.4f}.jpg")

        img, _, noise = generate_single(Gs, D, z=zx,
                                        compute_discriminator=False)

        cv.img = img

        cv += ph.text(x=2, y=-2, text=f"{delta:0.3f}")
        
        cv.save(f_save)       
        cv.resize(0.5).show(1)

