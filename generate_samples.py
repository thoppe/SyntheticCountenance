import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json

dim = 512

def load_model():
    print("Loading the model")

    import tensorflow as tf
    import sys
    import pickle

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=config)
    config.gpu_options.allow_growth = True

    f_model = 'model/karras2018iclr-celebahq-1024x1024.pkl'
    path_pg_gan_code = 'src/model/pggan/'

    sys.path.append(path_pg_gan_code)
    with open(f_model, 'rb') as FIN:
        G, D, Gs = pickle.load(FIN)

    return G, D, Gs

def compute_single(generator, discriminator, compute_discriminator=True):

    z = np.random.randn(dim)

    zf = z[None, :]
    dummy = np.zeros([z.shape[0], 0])

    img = generator.run(zf, dummy)

    if compute_discriminator:
        ds = discriminator.run(img)[0].ravel()[0]
    else:
        ds = None

    # [-1,1] => [0,255]
    img = np.clip(np.rint(
        (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC

    return img[0], ds, z

G, D, Gs = load_model()
save_dest_imgs = 'examples/imgs'
save_dest_info = 'examples/info'

os.system(f'mkdir -p {save_dest_imgs}')
os.system(f'mkdir -p {save_dest_info}')

for n in tqdm(range(10000)):
    img, ds, z = compute_single(Gs, D)

    f_save = os.path.join(save_dest_imgs, f"{n:06d}.jpg")
    P_img = Image.fromarray(img)
    P_img.save(f_save)

    info = {
        "z" : [float(_) for _ in z],
        "n" : n,
        "discriminator_score" : float(ds),         
    }

    f_save_info = os.path.join(save_dest_info, f"{n:06d}.json")
    with open(f_save_info, 'w') as FOUT:
        FOUT.write(json.dumps(info, indent=2))

    print(f_save)



'''
for n in tqdm(range(10000)):
    img, ds = compute_single(Gs, D)

    f_save = os.path.join(save_dest, f"{-ds:0.4f}.jpg")

    P_img = Image.fromarray(img)
    P_img.save(f_save)

    print(f_save)
'''
