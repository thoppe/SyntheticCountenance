import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json, glob
from src.GAN_model import load_GAN_model, generate_single
import dnnlib.tflib as tflib

n_samples = 10 ** 5
random_seed = 42

np.random.seed(random_seed)
G, D, Gs = load_GAN_model()
save_dest_imgs = "samples/images_noise"
#save_dest_info = "samples/latent_vectors"

os.system(f"mkdir -p {save_dest_imgs}")
#os.system(f"mkdir -p {save_dest_info}")

largest_known = glob.glob(os.path.join(save_dest_imgs, "*"))
largest_idx = 0

if len(largest_known):
    largest_known = os.path.basename(sorted(largest_known)[-1])
    largest_idx = int(largest_known.split(".")[0])

#logger.info(f"Starting generation at {largest_idx}")

noise_vars = [
    var for name, var in
    Gs.components.synthesis.vars.items() if name.startswith('noise')]

noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))


for i in range(43):
    z = np.random.randn(512)


# Observation: layer 4 is the hair wave!
n = 0
for k in tqdm(range(len(noise_vars))):
    print(f"Starting level {k}")
    for _ in range(10):

        f_save = os.path.join(save_dest_imgs, f"{n:08d}.jpg")
        #if os.path.exists(f_save):
        #    continue

        noise_index = k
        nx = noise_vars[noise_index]

        small_noise = np.random.standard_normal(nx.shape)
        tflib.set_vars({nx:small_noise})

        img, z, ds = generate_single(Gs, D, z=z, compute_discriminator=False)

        P_img = Image.fromarray(img)
        P_img.save(f_save)

        #f_save_info = os.path.join(save_dest_info, f"{n:08d}.npy")
        #np.save(f_save_info, z)
        n += 1
