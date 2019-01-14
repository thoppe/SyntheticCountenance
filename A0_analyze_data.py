import os, json, glob, h5py
import numpy as np
import pandas as pd

f_h5 = "samples/PGAN_attributes.h5"
h5 = h5py.File(f_h5, "r")

g = h5["image_attributes"]

df = pd.DataFrame(index=g["image_idx"][...])

cols = ["laplacian_variance", "laplacian_mean"]
for col in cols:
    df[col] = g[col][...]

key = "laplacian_variance"
# key ='laplacian_mean'

df = df.sort_values(key)

import pixelhouse as ph


# Laplacian_variance
# 250 is a good maximum
# Scary faces at < 10
# Malformed faces at < 17


for n in df.index[::1][::]:

    if np.random.uniform() < 0.1:
        continue

    f_img = f"samples/images/{n:06d}.jpg"
    if not os.path.exists(f_img):
        continue
    C = ph.Canvas()
    C.load(f_img)

    # text=str(np.round(df.loc[n, key]))
    # text=str(np.round(df.loc[n, key], 3))
    text = str(n)

    C += ph.text(x=2, y=-2, text=text, font_size=0.5)
    C.show()

    print(n)

# 662
