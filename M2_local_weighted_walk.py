import numpy as np
from tqdm import tqdm
from PIL import Image
import os, json, glob
import scipy.stats
from src.GAN_model import load_GAN_model, generate_single, logger


def interpolate(Z, frames_per=30):
    n = len(Z)
    T = np.linspace(0, n - 1, n * frames_per)

    sigma = np.random.uniform(low=0.5, high=0.75, size=[n,])
    G = [scipy.stats.norm(loc=k, scale=s) for k,s in zip(range(n), sigma)]

    for frame_idx, t in tqdm(enumerate(T), total=len(T)):
        weights = np.array([g.pdf(t) for g in G])

        # Need to wrap around weights for perfect loop
        weights += np.array([g.pdf(t + T.max()) for g in G])
        weights += np.array([g.pdf(t - T.max()) for g in G])

        # Set the strength of the weights to be unity
        weights /= weights.sum()

        zx = (Z.T * weights).sum(axis=1)
        zx /= np.sqrt((weights ** 2).sum())

        img, *_ = generate_single(Gs, D, z=zx, compute_discriminator=False)

        f_save = os.path.join(save_dest, f"{frame_idx:08d}.jpg")
        scipy.misc.imsave(f_save, img)



def load_z(n):
    if os.path.exists(n):
        f_npy = n
    else:
        f_npy = f"samples/latent_vectors/{n:06d}.npy"

    assert(os.path.exists(f_npy))
    z = np.load(f_npy)

    name = os.path.basename(f_npy).split('.')[0]
    return z, name
        

if __name__ == "__main__":

    np.random.seed(43)
    
    z0, name = load_z(360)
    z0, name = load_z("saved/AliS.npy")

    G, D, Gs = load_GAN_model()

    save_dest = f"motion/transport_walk/{name}"
    save_dest_ref = f"motion/transport_walk/{name}_reference"

    os.system(f"rm -rf {save_dest} {save_dest_ref}")
    os.system(f"mkdir -p {save_dest} {save_dest_ref}")

    epsilon = 0.30
    n_transitions = 20

    # Vicinity sampling around the center
    delta = epsilon * np.random.standard_normal(size=(n_transitions - 1, 512))

    # Cherry-pick some bad samples away
    for k in [12, 6, 0, 6, 6]:
        delta[k] = np.random.standard_normal(size=(512,))
    
    Z = z0 + delta

    # Build the reference samples
    logger.debug(f"Building the reference samples in {save_dest_ref}")
    for n, zx in tqdm(enumerate(Z)):
        img, *_ = generate_single(Gs, D, z=zx, compute_discriminator=False)
        f_save = os.path.join(save_dest_ref, f"{n:08d}.jpg")
        scipy.misc.imsave(f_save, img)
    
    # Add the starting point back on for smooth loops 
    Z = np.vstack([Z, Z[0]])

    # Scale factor (see arXiv:1711.01970 Table 1)
    Z /= np.sqrt(1 + epsilon ** 2)

    interpolate(Z, frames_per=30)
