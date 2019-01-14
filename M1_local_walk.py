import pixelhouse as ph
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json
from src.GAN_model import load_GAN_model, generate_single


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


def spherical_bezier(q0, q1, q2, q3):
    # Dev note: Sadly, this doesn't work well (goes too far into the surface)

    # Adapted from
    # Quaternions and SLERP, Verena Elisabeth Kremer

    # http://citeseerx.ist.psu.edu/viewdoc/download
    # ?doi=10.1.1.479.6134&rep=rep1&type=pdf

    """
    def bisect(p, q):
        total = p+q
        return total/np.linalg.norm(total)

    def doubleq(p,q):
        return 2*(p*q)*q -p

    an = bisect(doubleq(q0, q1), q2)
    an1 = bisect(doubleq(q1, q2), q3)
    
    bn = doubleq(an, q1)
    bn1= doubleq(an1, q2)

    bn1 /= np.linalg.norm(bn1)
    an /= np.linalg.norm(an)
    
    p00 = q1
    p10 = an
    p20 = bn1
    p30 = q2
    """

    p00 = q0

    # Controls how close to the original points we want
    tx = 0.75

    p10 = SLERP(q0, q1)(tx)
    p20 = SLERP(q2, q3)(1 - tx)

    # p10 = (tx*q0+(1-tx)*q1) #/ np.sqrt(tx**2+(1-tx)**2)
    # p20 = (tx*q3+(1-tx)*q2) #/ np.sqrt(tx**2+(1-tx)**2)

    p30 = q3

    def func(t):
        p01 = SLERP(p00, p10)(t)
        p11 = SLERP(p10, p20)(t)
        p21 = SLERP(p20, p30)(t)
        p02 = SLERP(p01, p11)(t)
        p12 = SLERP(p11, p21)(t)
        p03 = SLERP(p02, p12)(t)
        return p03

    return func


def interpolate(z0, z1, z2, z3, frame_idx=0, frames_per=30):
    # def interpolate(z0, z1, frame_idx=0, frames_per=30):

    # ZS = SLERP(z0, z1)
    ZS = spherical_bezier(z0, z1, z2, z3)

    ITR = np.linspace(0, 1, frames_per + 1)[:-1]

    motion = ph.motion.easeInOutQuad(0, 1)
    C = ph.Canvas(1024, 1024)

    for t in ITR:

        # Easing motion, useful for regular SLERP
        a = motion(t)

        # Linear motion, better for spherical_bezier
        # a = t

        img, *_ = generate_single(Gs, D, z=ZS(a), compute_discriminator=False)
        C.img[:, :, :3] = img

        # C += ph.text(x=0, y=3, text=text, font_size=0.60)
        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")

        C.save(f_save)
        frame_idx += 1

        print(f_save)

    return frame_idx


G, D, Gs = load_GAN_model()

target_idx = 360
# target_idx = 165
# target_idx = 662  # blue goth
# target_idx = 17048
# target_idx = 24498
# target_idx = 44751


save_dest = f"motion/local_walk/{target_idx:06d}"
os.system(f"rm -rf {save_dest}")
os.system(f"mkdir -p {save_dest}")

f_npy = f"samples/latent_vectors/{target_idx:06d}.npy"
z = np.load(f_npy)

epsilon = 0.20
n_transistions = 6

# Set a series of local walks around the central latent vector
# Vicinity sampling
Z = []
for _ in range(4 * n_transistions):
    zx = z + epsilon * np.random.randn(512)

    # Scale factor (see arXiv:1711.01970 Table 1)
    zx /= np.sqrt(1 + epsilon ** 2)
    Z.append(zx)


# fk = 0
# for z0, z1 in zip(Z, Z[1:]):
#    fk = interpolate(z0, z1, fk, frames_per=30)


fk = 0
for i in range(4, len(Z), 3):
    fk = interpolate(Z[i - 4], Z[i - 3], Z[i - 2], Z[i - 1], fk, frames_per=30)
