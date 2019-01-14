import numpy as np
import cv2, os, json
from imutils import face_utils
from src.pipeline import Pipeline


def compute(f_image, f_json):

    img = cv2.imread(f_image)
    item = {}

    lap = cv2.Laplacian(img, cv2.CV_64F)
    item["laplacian_mean"] = lap.mean()
    item["laplacian_variance"] = lap.var()

    js = json.dumps(item, indent=2)

    with open(f_json, "w") as FOUT:
        FOUT.write(js)


P = Pipeline(
    load_dest="samples/images",
    save_dest="samples/image_attributes",
    new_extension="json",
)(compute, -1)
