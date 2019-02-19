from src import pipeline
from src.logger import logger

import numpy as np

import pandas as pd
from tqdm import tqdm
import os, json, glob
import cv2
import random
from lapjv import lapjv
from scipy.spatial.distance import cdist

# Read half from the left and half from the right
image_grid = 14
assert(image_grid%2==0)
random_seed = 123


def fit_to_grid(IMG, X_2d, n, m, out_res=224):
    grid = np.dstack(np.meshgrid(
        np.linspace(0, 1, n),
        np.linspace(0, 1, m))).reshape(-1, 2)

    
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    return col_asses


def generate_tsne(activations, perplexity=50, tsne_iter=5000):
    # Run tSNE in parallel if the proper library is installed

    args = {
        "perplexity" : perplexity,
        "n_components" : 2,
        "n_iter" : tsne_iter,
        "init" : "random",    
    }

    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
        args["n_jobs"] = -1
    except ModuleNotFoundError:
        import warnings
        warnings.warn("Using slow sklearn TSNE")
        from sklearn.manifold import TSNE

    X = TSNE(**args).fit_transform(np.array(activations))
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X


ordering = None

def compute(f0, f1):
    global ordering

    person_idx = int(os.path.basename(f0).split('_')[0])
    frame_idx = int(os.path.basename(f0).split('_')[1].split('.')[0])

    f1 = os.path.join(os.path.dirname(f1), f"{frame_idx:06d}.jpg")
    #if os.path.exists(f1):
    #    return False

    #img = cv2.imread(f0)
    
    n_imgs = (image_grid**2)//2
    
    if person_idx != 0:
        return False

    left_img = sorted(glob.glob(os.path.join(os.path.dirname(f0),
        f'*_{frame_idx:06d}*.*')))

    right_img = sorted(glob.glob(os.path.join(os.path.dirname(f0).replace(
        'left', 'right'),
        f'*_{frame_idx:06d}*.*')))

    # Adhoc removal
    left_img = [x for x in left_img if "0065_" not in x]
    right_img = [x for x in right_img if "0065_" not in x]
    
    print(f"Needed {n_imgs}, found {len(left_img)}")

    assert(n_imgs <= len(left_img))
    assert(n_imgs <= len(right_img))

    left_img = left_img[:n_imgs]
    right_img = right_img[:n_imgs]

    '''
    if ordering is None:
        print("ORDERING!")

        activations = np.array(
            [cv2.imread(f).flatten().astype(np.float)/255 for f in left_img])

        X = generate_tsne(activations)
        n = image_grid//2
        m = image_grid
        ordering = fit_to_grid(left_img, X, n , m)

    left_img = np.array(left_img)[ordering]
    right_img = np.array(right_img)[ordering]
    '''


    sample_img = cv2.imread(left_img[0])
    height, width = sample_img.shape[:2]

    canvas_width = image_grid*width
    canvas_height = image_grid*height

    img = np.zeros(
        (canvas_height, canvas_width, 3), dtype=sample_img.dtype)

    ITR = iter(left_img)
    for i in range(image_grid//2):
        for j in range(image_grid):
            x0, y0 = i*width, j*height
            f_img = next(ITR)
            small = cv2.imread(f_img)
            img[y0:y0+height, x0:x0+width, :] = small

    ITR = iter(right_img)
    for i in range(image_grid//2)[::-1]:
        for j in range(image_grid):
            x0, y0 = i*width, j*height
            x0 += canvas_width//2
            f_img = next(ITR)
            small = cv2.imread(f_img)
            img[y0:y0+height, x0:x0+width, :] = small

            tc = 2
    color = (255,255,255)
    color = (15,)*3
    #color = img.mean(axis=0).mean(axis=0).astype(np.uint8).tolist()

    
    for i in range(image_grid):
        x0 = i*width
        cv2.line(img, (x0,0), (x0, canvas_height), color=color, thickness=tc)

    for j in range(image_grid):
        y0 = j*height
        cv2.line(img, (0,y0), (canvas_width, y0), color=color, thickness=tc)
            
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #exit()

    cv2.imwrite(f1, img)

    

PIPE = pipeline.Pipeline(
    #load_dest = 'working_tripgan/left_eye/',
    #save_dest = 'working_tripgan/stacked/',

    load_dest = 'tripgan/left_eye/',
    save_dest = 'tripgan/stacked/',
    old_extension = 'jpg',
    new_extension = 'jpg',
    shuffle=False
)(compute, 1)
