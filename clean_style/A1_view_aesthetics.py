import glob, os
import pixelhouse as ph
import numpy as np
import pandas as pd
from tqdm import tqdm

F_NPY = glob.glob('data/AS_score/*.npy')[::1]
load_dest = 'data/images/'

data = []
for f in tqdm(F_NPY):
    f_img = os.path.join(load_dest, os.path.basename(f).replace('.npy','.jpg'))
    scores = np.load(f)

    x = scores*range(len(scores))
    data.append({
        "f_img":f_img,
        #"mu": np.average(range(len(scores)), weights=scores),
        "mu": x.mean(),
        "std":x.std(),
        "scores":scores,
    })

key = 'std'
    
df = pd.DataFrame(data).sort_values(key, ascending=False)

for _,row in df.iterrows():
    print(f"Showing {row.f_img}, {row.scores}")
    cv = ph.load(row.f_img)
    cv += ph.text(x=2.5, y=-2.25, text=f"{row['mu']:0.2f}")
    cv += ph.text(x=2.5, y=-3, text=f"{row['std']:0.2f}")
    cv.show()

