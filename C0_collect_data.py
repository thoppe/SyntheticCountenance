import numpy as np
import os, json, glob, h5py
import pandas as pd
from tqdm import tqdm

# Load the image attributes

def collect_attributes():

    F_ATTRIBUTES = sorted(glob.glob("samples/image_attributes/*.json"))
    data = []
    for f in tqdm(F_ATTRIBUTES):
        with open(f) as FIN:
            js = json.loads(FIN.read())
            js['name'] = int(os.path.basename(f).split('.')[0])
        data.append(js)
    df = pd.DataFrame(data).sort_values('name').set_index('name')

    label = 'image_attributes'

    if label in h5:
        del h5[label]

    g = h5.require_group(label)

    g['image_idx'] = df.index.values
    for key in df:
        g[key] = df[key].values


def collect_vectors(label):
    print(f"Collecting {label}")
    F_VECTORS = sorted(glob.glob(
        f"samples/{label}/*.npy"))

    X = []
    image_idx = []
    
    for f in tqdm(F_VECTORS):
        X.append(np.load(f))
        image_idx.append(int(os.path.basename(f).split('.')[0]))

    if label in h5:
        del h5[label]

    g = h5.require_group(label)
    g['image_idx'] = image_idx
    g['data'] = X


f_h5 = 'samples/PGAN_attributes.h5'
if not os.path.exists(f_h5):
    with h5py.File(f_h5, 'w'):
        pass
h5 = h5py.File(f_h5, 'r+')
    
if __name__ == "__main__":
    collect_attributes()
    collect_vectors('face_vectors')
    collect_vectors('latent_vectors')
    collect_vectors('keypoints_68')
    collect_vectors('keypoints_5')




