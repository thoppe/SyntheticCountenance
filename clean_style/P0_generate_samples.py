import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single

n_samples = 10 ** 5
random_seed = 42

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()
save_dest_imgs = "data/images"
save_dest_info = "data/latent_vectors"

os.system(f"mkdir -p {save_dest_imgs}")
os.system(f"mkdir -p {save_dest_info}")

dim = 512
known_images = set(glob.glob(os.path.join(save_dest_imgs, "*")))

for n in tqdm(range(0, n_samples)):

    z = np.random.randn(dim)

    f_save = os.path.join(save_dest_imgs, f"{n:08d}.jpg")
    if f_save in known_images:
        continue

    img, z, ds = generate_single(Gs, D, z=z, compute_discriminator=False)

    P_img = Image.fromarray(img)
    P_img.save(f_save)

    f_save_info = os.path.join(save_dest_info, f"{n:08d}.npy")
    np.save(f_save_info, z)
