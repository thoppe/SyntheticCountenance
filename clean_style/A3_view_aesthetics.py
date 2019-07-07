import glob, os, json
import pixelhouse as ph
import numpy as np
import pandas as pd
from tqdm import tqdm

F_JSON = glob.glob("data/Aesthetics_score/*.json")
load_dest = "data/images/"

data = []
for f in tqdm(F_JSON):
    with open(f) as FIN:
        js = json.load(FIN)
        data.append(js)

known_keys = (
    "Aesthetic",
    "BalancingElement",
    "ColorHarmony",
    "Content",
    "DoF",
    "Light",
    "MotionBlur",
    "Object",
    "Repetition",
    "RuleOfThrids",
    "Symmetry",
    "VividColor",
)

size = 0.25

df = pd.DataFrame(data)


for key in known_keys:

    df = df.sort_values(key, ascending=False)

    good = []
    for _, row in df[:6].iterrows():
        print(f"Showing {row.f_img}")
        cv = ph.load(row.f_img)
        cv += ph.text(x=2.5, y=-3, text=f"{row[key]:0.2f}")
        good.append(cv.resize(size))

    good = ph.hstack(good)

    bad = []
    for _, row in df[::-1][:6].iterrows():
        print(f"Showing {row.f_img}")
        cv = ph.load(row.f_img)
        cv += ph.text(x=2.5, y=-3, text=f"{row[key]:0.2f}")
        bad.append(cv.resize(size))

    bad = ph.hstack(bad)
    img = ph.vstack([good, bad])

    img += ph.text(x=0, y=0, font_size=0.2, text=f"{key}")

    img.save(f"figures/{key}.png")
    img.show(1)
