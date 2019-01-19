import numpy as np
import cv2, os, json
from PIL import Image
from src.pipeline import Pipeline

new_image_dim = 256

def compute(f_image, f_image_out):

    img = cv2.imread(f_image)
    item = {}

    dim = (new_image_dim, new_image_dim)
    img = Image.open(f_image).resize(dim, Image.ANTIALIAS)
    img.save(f_image_out)

P = Pipeline(
    load_dest="samples/images",
    save_dest=f"samples/image_resized_{new_image_dim}",
    new_extension="jpg",
)(compute, -1)
