import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json

from src.GAN_model import load_GAN_model, generate_single

max_images = 10**6

G, D, Gs = load_GAN_model()
save_dest_imgs = 'samples/images'
save_dest_info = 'samples/latent_vectors'

os.system(f'mkdir -p {save_dest_imgs}')
os.system(f'mkdir -p {save_dest_info}')

for n in tqdm(range(max_images)):
    
    f_save = os.path.join(save_dest_imgs, f"{n:06d}.jpg")
    if os.path.exists(f_save):
        continue
    
    img, z, ds = generate_single(Gs, D)
    
    P_img = Image.fromarray(img)
    P_img.save(f_save)

    f_save_info = os.path.join(save_dest_info, f"{n:06d}.npy")
    np.save(f_save_info, z)
