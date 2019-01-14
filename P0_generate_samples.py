import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single

n_samples = 10 ** 2

G, D, Gs = load_GAN_model()
save_dest_imgs = "samples/images"
save_dest_info = "samples/latent_vectors"

os.system(f"mkdir -p {save_dest_imgs}")
os.system(f"mkdir -p {save_dest_info}")

largest_known = glob.glob(os.path.join(save_dest_imgs, "*"))
largest_idx = 0

if len(largest_known):
    largest_known = os.path.basename(sorted(largest_known)[-1])
    largest_idx = int(largest_known.split(".")[0])


print("Starting generation")

for n in tqdm(range(largest_idx, n_samples)):

    f_save = os.path.join(save_dest_imgs, f"{n:06d}.jpg")
    if os.path.exists(f_save):
        continue

    img, z, ds = generate_single(Gs, D, compute_discriminator=False)

    P_img = Image.fromarray(img)
    P_img.save(f_save)

    f_save_info = os.path.join(save_dest_info, f"{n:06d}.npy")
    np.save(f_save_info, z)
