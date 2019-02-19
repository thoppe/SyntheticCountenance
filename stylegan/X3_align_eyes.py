from src import pipeline
from src.logger import logger

import numpy as np
from tqdm import tqdm
import cv2
import os, json, glob, tempfile

n_expected_images = 90


def compute(f0, f1):
    src_dst, name = os.path.dirname(f0), os.path.basename(f0)
    person_idx = int(name.split('_')[0])
    frame_idx = int(name.split('_')[1].split('.')[0])

    # Start only on the first frame
    if frame_idx != 0:
        return False

    final_dst = os.path.dirname(f1)

    matching = sorted(
        glob.glob(os.path.join(src_dst, f"{person_idx:04d}_*.png")))

    if len(matching) != n_expected_images:
        logger.warning(f"Not enough images for {f0}")
        return False

    matching = [os.path.abspath(x) for x in matching][:]

    current_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as build_dst:
        os.chdir(build_dst)

        f_img = ' '.join(matching)
        cmd = f'align_image_stack -m --use-given-order -a aligned_ {f_img}'
        os.system(cmd)


        created = glob.glob('aligned_*.tif')
        for k in range(len(created)):
            f_aligned = f'aligned_{k:04d}.tif'
            img = cv2.imread(f_aligned, cv2.IMREAD_UNCHANGED)
            f_save = os.path.join(current_dir, final_dst, f"{person_idx:04d}_{k:06d}.png")
            cv2.imwrite(f_save, img)
            #print(img.shape, f_save)
            #cv2.imshow('image', img)
            #cv2.waitKey(0)

    os.chdir(current_dir)
    
    
PIPE = pipeline.Pipeline(
    load_dest = 'monstergan/left_eye/',
    save_dest = 'monstergan/left_eye_aligned/',
    old_extension = 'png',
    new_extension = 'png',
    shuffle=True,
)(compute, -1)

PIPE = pipeline.Pipeline(
    load_dest = 'monstergan/right_eye/',
    save_dest = 'monstergan/right_eye_aligned/',
    old_extension = 'png',
    new_extension = 'png',
    shuffle=True,
)(compute, -1)
