import numpy as np
import os, json, glob, h5py
import pandas as pd
from tqdm import tqdm

f_h5 = 'samples/PGAN_attributes.h5'
if not os.path.exists(f_h5):
    with h5py.File(f_h5, 'w'):
        pass
h5 = h5py.File(f_h5, 'r+')

# Load the image attributes

F_ATTRIBUTES = sorted(glob.glob("samples/image_attributes/*.json"))
data = []
for f in tqdm(F_ATTRIBUTES):
    with open(f) as FIN:
        js = json.loads(FIN.read())
        js['name'] = int(os.path.basename(f).split('.')[0])
    data.append(js)
df = pd.DataFrame(data).sort_values('name').set_index('name')

del h5['image_attributes']
g = h5.require_group("image_attributes")

g['image_idx'] = df.index.values
for key in df:
    g[key] = df[key].values




