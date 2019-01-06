import os, json, glob
from tqdm import tqdm
import random
from face_classification.classify_image import classify

F_JPG = glob.glob('../examples/imgs/*.jpg')
random.shuffle(F_JPG)

for f in tqdm(F_JPG):
    f_json = f.replace('/imgs/', '/info/').replace('.jpg', '.json')

    if not os.path.exists(f_json):
        print(f"Removing {f}")
        os.remove(f)
        continue
    
    with open(f_json) as FIN:
        js = json.loads(FIN.read())

    if 'faces' in js:
        continue

    js['faces'] = classify(f)

    with open(f_json, 'w') as FOUT:
        text = json.dumps(js)
        FOUT.write(text)

    print(f)
