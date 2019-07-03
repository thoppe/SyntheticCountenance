import glob, os
import pixelhouse as ph
import numpy as np
import pandas as pd
from tqdm import tqdm

F_NPY = glob.glob('samples/AS_images/*.npy')[:]
load_dest = 'samples/images/'

data = []
for f in tqdm(F_NPY):
    f_img = os.path.join(load_dest, os.path.basename(f).replace('.npy','.jpg'))
    scores = np.load(f)
    data.append({
        "f_img":f_img,
        "mu": np.average(range(len(scores)), weights=scores),
    })
    
df = pd.DataFrame(data).sort_values("mu")#, ascending=False)

for _,row in df.iterrows():
    cv = ph.load(row.f_img)
    cv += ph.text(x=2.5, y=-3, text=f"{row.mu:0.2f}")
    cv.show()

