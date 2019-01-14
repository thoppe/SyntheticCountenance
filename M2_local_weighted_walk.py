import pixelhouse as ph
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json
import scipy.stats
from src.GAN_model import load_GAN_model, generate_single


def interpolate(Z, frames_per=30):
    n = len(Z)
    T = np.linspace(0, n - 1, n * frames_per)

    sigma = 0.50
    G = [scipy.stats.norm(loc=k, scale=sigma) for k in range(n)]

    motion = ph.motion.easeInOutQuad(0, 1)
    C = ph.Canvas(1024, 1024)

    for frame_idx, t in enumerate(T):
        weights = np.array([g.pdf(t) for g in G])

        # Need to wrap around weights for perfect loop
        weights += np.array([g.pdf(t + T.max()) for g in G])
        weights += np.array([g.pdf(t - T.max()) for g in G])

        # Normal
        weights /= weights.sum()

        zx = (Z.T * weights).sum(axis=1)
        zx /= np.sqrt((weights ** 2).sum())

        img, *_ = generate_single(Gs, D, z=zx, compute_discriminator=False)
        C.img[:, :, :3] = img

        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")
        C.save(f_save)

        print(f_save)

    """
    for t in ITR:

        # Easing motion, useful for regular SLERP
        a = motion(t)

        # Linear motion, better for spherical_bezier
        #a = t        

        frame_idx += 1

        print(f_save)
    """


G, D, Gs = load_GAN_model()

target_idx = 360
# target_idx = 165
target_idx = 662  # blue goth
# target_idx = 17048
# target_idx = 44751


save_dest = f"motion/transport_walk/{target_idx:06d}"
os.system(f"rm -rf {save_dest}")
os.system(f"mkdir -p {save_dest}")

f_npy = f"samples/latent_vectors/{target_idx:06d}.npy"
z = np.load(f_npy)

epsilon = 0.40
n_transitions = 20

# Vicinity sampling around the center
Z = z + epsilon * np.random.standard_normal(size=(n_transitions - 1, 512))

# Add the starting point back on
Z = np.vstack([Z, Z[0]])

# Scale factor (see arXiv:1711.01970 Table 1)
Z /= np.sqrt(1 + epsilon ** 2)

interpolate(Z, frames_per=30)
