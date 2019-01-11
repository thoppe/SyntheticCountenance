import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os, json
import tensorflow as tf

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

G, D, Gs = load_model()

'''
    # [-1,1] => [0,255]
    img = np.clip(np.rint(
        (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC
'''

image_dim = 1024


minibatch_size = 2

latents = tf.random_normal([minibatch_size] + Gs.input_shapes[0][1:])
labels = training_set.get_random_labels_tf(minibatch_size)
fake_images_out = Gs.get_output_for(latents, labels, is_training=False)


#print(Gs.list_layers())
#help(G)
#G_paper_1/latents_in:0
#img = G(z_input)



