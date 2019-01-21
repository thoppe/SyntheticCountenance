import numpy as np
import cv2, os, json
from tqdm import tqdm
from PIL import Image
from src.pipeline import Pipeline

new_image_dim = 256

def compute(f_image, f_image_out):

    img = cv2.imread(f_image)
    item = {}

    dim = (new_image_dim, new_image_dim)
    img = Image.open(f_image).resize(dim, Image.ANTIALIAS)
    img.save(f_image_out)

save_dest = f"samples/image_resized_{new_image_dim}"
    
P = Pipeline(
    load_dest="samples/images",
    save_dest=save_dest,
    new_extension="jpg",
)(compute, -1)

# Now build a quick lookup table
f_train_table = 'samples/PGAN_small_images.h5'

import h5py, glob

F_IMG = tqdm(glob.glob(os.path.join(save_dest, '*.jpg')))

IMG, Z = [], []

for f in F_IMG:

    name = os.path.basename(f).split('.')[0]
    f_npy = os.path.join(f'samples/latent_vectors/{name}.npy')

    if not os.path.exists(f_npy):
        print(f"Skipping {f_npy}")
        continue

    Z.append(np.load(f_npy))
    IMG.append(np.array(Image.open(f)))


Z = np.array(Z)
IMG = np.array(IMG)

with h5py.File(f_train_table, 'w') as h5:
    h5['Z'] = Z
    h5['IMG'] = IMG
