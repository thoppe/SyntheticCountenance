import pixelhouse as ph
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json

dim = 512

# Use this?
# https://towardsdatascience.com/do-gans-really-model-the-true-data-distribution-or-are-they-just-cleverly-fooling-us-d08df69f25eb


def load_model():
    print("Loading the model")

    import tensorflow as tf
    import sys
    import pickle

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.InteractiveSession(config=config)
    config.gpu_options.allow_growth = True

    f_model = "model/karras2018iclr-celebahq-1024x1024.pkl"
    path_pg_gan_code = "src/model/pggan/"

    sys.path.append(path_pg_gan_code)
    with open(f_model, "rb") as FIN:
        G, D, Gs = pickle.load(FIN)

    return G, D, Gs


def compute_single(generator, discriminator, z=None, compute_discriminator=False):

    if z is None:
        z = np.random.randn(dim)

    zf = z[None, :]
    dummy = np.zeros([z.shape[0], 0])

    img = generator.run(zf, dummy)

    if compute_discriminator:
        ds = discriminator.run(img)[0].ravel()[0]
    else:
        ds = None

    # [-1,1] => [0,255]
    img = np.clip(np.rint((img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC

    return img[0]


class SLERP:
    def __init__(self, z0, z1):
        theta = z0.dot(z1)
        theta /= np.linalg.norm(z0)
        theta /= np.linalg.norm(z1)
        self.theta = theta
        self.z0, self.z1 = z0, z1

    def __call__(self, t):
        st = np.sin(self.theta)
        s0 = np.sin((1 - t) * self.theta) / st
        s1 = np.sin(t * self.theta) / st
        return s0 * self.z0 + s1 * self.z1


def interpolate(k0, k1, frame_idx=0, frames_per=30):

    load_info = "examples/info"

    f_info = os.path.join(load_info, f"{k0:06d}.json")
    with open(f_info) as FIN:
        z0 = np.array(json.loads(FIN.read())["z"])

    f_info = os.path.join(load_info, f"{k1:06d}.json")
    with open(f_info) as FIN:
        z1 = np.array(json.loads(FIN.read())["z"])

    ZS = SLERP(z0, z1)
    ITR = np.linspace(0, 1, frames_per + 1)[:-1]

    motion = ph.motion.easeInBack(0, 1)

    C = ph.Canvas(1024, 1024)

    for t in ITR:
        # a = motion(t)
        a = t

        img = compute_single(Gs, D, z=ZS(a))

        if t == 0:
            # Need to precompute and filter against this
            val = cv2.Laplacian(img, cv2.CV_64F).var()

        C.img[:, :, :3] = img

        text = f"{frame_idx}"
        text = f"{val:0.4f}"

        C += ph.text(x=0, y=3, text=text, font_size=0.60)
        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")

        C.save(f_save)

        # P_img = Image.fromarray(img)
        # P_img.save(f_save)

        frame_idx += 1

        print(f_save)

    return frame_idx


G, D, Gs = load_model()

save_dest = "motion/happy_sad"
os.system(f"rm -rf {save_dest}")
os.system(f"mkdir -p {save_dest}")


f_info = "analysis/latent_gender_and_emotion_training.csv"
df = pd.read_csv(f_info)
df = df[(df.happy > 0.9) | (df.sad > 0.9)]


n_transistions = 20

K = []
for _ in range(n_transistions):
    K.append(np.random.choice(df.loc[df.happy > 0.9, "n"]))
    K.append(np.random.choice(df.loc[df.sad > 0.9, "n"]))

K.append(np.random.choice(df.loc[df.happy > 0.9, "n"]))


fk = 0
for k0, k1 in zip(K, K[1:]):
    fk = interpolate(k0, k1, fk, frames_per=30)
