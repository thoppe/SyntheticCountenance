import glob, os, json
import pixelhouse as ph
import numpy as np
import pandas as pd
from tqdm import tqdm

F_JSON = glob.glob('data/MEMNET_score/*.json')
load_dest = 'data/images/'

data = []
for f in tqdm(F_JSON):
    with open(f) as FIN:
        data.append(json.load(FIN))

key = 'memnet_score'
  
#df = pd.DataFrame(data).sort_values(key, ascending=True)
df = pd.DataFrame(data).sort_values(key, ascending=False)

for _,row in df.iterrows():
    print(f"Showing {row.f_img}")
    cv = ph.load(row.f_img)
    cv += ph.text(x=2.5, y=-3, text=f"{row[key]:0.2f}")
    cv.show()

