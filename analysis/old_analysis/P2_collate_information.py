import pandas as pd
from tqdm import tqdm
import numpy as np
import h5py
import glob, json, os

f_h5 = 'latent_gender_and_emotion_training.h5'
f_save_csv = 'latent_gender_and_emotion_training.csv'

F_INFO = glob.glob('../examples/info/*.json')

data = []
emotions = None
Z = []
keypoints = []

for f_json in tqdm(F_INFO):

            
    with open(f_json) as FIN:
        try:
            js = json.loads(FIN.read())
        except json.decoder.JSONDecodeError:
            print(f"Problem with json, removing {f_json}")
            os.remove(f_json)            
            continue

    if 'faces' not in js:
        continue

    if 'keypoints' not in js:
        continue

    kp = js['keypoints']
    if len(kp) != 68:
        continue

    faces = js['faces']

    if len(faces) != 1:
        continue

    face = faces[0]

    item = {'n':js['n']}
    item['f_image'] = f_json.replace(
        '/info/', '/imgs/').replace('.json', '.jpg')
    
    item.update(dict(
        zip(face['gender_labels'].values(), face['gender_vector'][0])))

    item.update(dict(
        zip(face['emotion_labels'].values(), face['emotion_vector'][0])))

    emotions = list(face['emotion_labels'].values())
    data.append(item)
    Z.append(js['z'])

    keypoints.append(np.array(kp))
    

df = pd.DataFrame(data).set_index('n')
df.to_csv(f_save_csv)

with h5py.File(f_h5, 'w') as h5:
    h5['Z'] = Z
    h5['keypoints'] = keypoints
    h5['n'] = df.index.values
    
    for key in ['man', 'woman'] + emotions:
        h5[key] = df[key].values

print(f"Found {len(Z)} reasonable images")
exit()


df = df.sort_values('man')

import pixelhouse as ph
img = ph.Canvas()

dfx = pd.concat([df[:3], df[-3:]])
#for f_img, val in zip(dfx.f_image, dfx.man):
#    img.load(f_img)
#    img += ph.text(x=2, y=-2, text=f'man {val:0.2f}', font_size=.2)
#    img.show()

for emotion in emotions:
    dfx = df.sort_values(emotion, ascending=False)[:3]
    for f_img, val in zip(dfx.f_image, dfx[emotion]):
        img.load(f_img)
        img += ph.text(x=2, y=-2, text=f'{emotion} {val:0.2f}', font_size=.3)
        img.show()

    


        
