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

def compute_single(
        generator, discriminator, z=None, compute_discriminator=True):

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
    img = np.clip(np.rint(
        (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC

    return img[0]


k0 = 10
k1 = 938
load_info = 'examples/info'

f_info = os.path.join(load_info, f"{k0:06d}.json")
with open(f_info) as FIN:
    z0 = np.array(json.loads(FIN.read())['z'])


f_info = os.path.join(load_info, f"{k1:06d}.json")
with open(f_info) as FIN:
    z1 = np.array(json.loads(FIN.read())['z'])


#import pixelhouse as ph
#C = ph.Canvas(1024, 1024)

G, D, Gs = load_model()

save_dest0 = 'motion/linear'
save_dest1 = 'motion/slerp/'

os.system(f'mkdir -p {save_dest0}')
os.system(f'mkdir -p {save_dest1}')

ITR = tqdm(np.linspace(0, 1, 100))

theta = z0.dot(z1)
theta /= np.linalg.norm(z0)
theta /= np.linalg.norm(z1)

for n, t in enumerate(ITR):

    s0 = np.sin((1-t)*theta)/np.sin(theta)
    s1 = np.sin(t*theta)/np.sin(theta)
    z_slerp = s0*z0 + s1*z1
    
    img = compute_single(Gs, D, z=z_slerp)
    
    f_save = os.path.join(save_dest1, f"{n:04d}.jpg")
    P_img = Image.fromarray(img)
    P_img.save(f_save)
    

    z_blend = t*z1 + (1-t)*z0
    img = compute_single(Gs, D, z=z_blend)
    
    f_save = os.path.join(save_dest0, f"{n:04d}.jpg")
    P_img = Image.fromarray(img)
    P_img.save(f_save)


'''
img = compute_single(Gs, D, z=z0)
C.img[:,:,:3] = img
C.show()

img = compute_single(Gs, D, z=z1)
C.img[:,:,:3] = img
C.show()
'''


