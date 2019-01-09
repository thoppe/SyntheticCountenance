import numpy as np
import pixelhouse as ph
import sys
import pickle
import tensorflow as tf
import time

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)
f_model = 'model/karras2018iclr-celebahq-1024x1024.pkl'
path_pg_gan_code = 'src/model/pggan/'

sys.path.append(path_pg_gan_code)
with open(f_model, 'rb') as FIN:
    G, D, Gs = pickle.load(FIN)


def generate_image(Gs):
    N = Gs.input_shapes[0][1]
    z = np.random.randn(N)
    z = z[None, :]
    dummy = np.zeros([z.shape[0], 0])
    img = Gs.run(z, dummy)

    score = D.run(img)[0].ravel()[0]

    # [-1,1] => [0,255]
    img = np.clip(np.rint(
        (img + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)

    img = img.transpose(0, 2, 3, 1)  # NCHW => NHWC
    
    return img[0], score


class LiveAnimation(ph.Animation):

    def show(self, delay=1):
        while True:
            img = self.render(0)
            self.has_rendered[0] = False
            img.show(delay=delay)
            time.sleep(1)


class GAN_viewer(ph.Artist):

     def __init__(self):
        pass

     def draw(self, cvs, t=0):
        img, score = generate_image(Gs)
        cvs.img[:, :, :3] = img
        cvs += ph.text(x=0, y=2, text=f"{score:0.2f}")


A = LiveAnimation(1024, 1024)
A += GAN_viewer()
A.show()
