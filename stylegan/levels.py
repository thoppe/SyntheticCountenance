import cv2
import numpy as np
from imutils import face_utils
import imutils, dlib, os

class FACE_FINDER:
    def __init__(self):

        model_dest = "model/dlib"
        self.shape68_pred = dlib.shape_predictor(
            f"{model_dest}/shape_predictor_68_face_landmarks.dat"
        )
        self.detector = dlib.get_frontal_face_detector()
        
    def compute_keypoints(self, img, bbox):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape68 = self.shape68_pred(gray, bbox)
        keypoints = np.array(face_utils.shape_to_np(shape68))

        return keypoints

    def compute_bbox(self, img, n_upsample=0):
        faces = self.detector(img, n_upsample)
        if len(faces) != 1:
            logger.warning(f"Found {len(faces)} faces in image! Expected one")
        return faces[0]

    def convex_hull(self, img):
        bbox = self.compute_bbox(img)
        pts = self.compute_keypoints(img, bbox)
        hull = cv2.convexHull(pts).squeeze()

        #kernel = np.ones((5,5),np.uint8)
        #expand = cv2.dilate(hull, kernel, iterations = 5)
        #return expand
    
        return hull

    def blur_mask(self, img, mask_blur=201):
        h, w = img.shape[:2]
        
        hull = self.convex_hull(img)
        mask = np.zeros((h, w)).astype(np.uint8)
        mask = cv2.drawContours(mask, [hull], 0, 255, -1)
        mask = mask.astype(bool)

        mask = 255*mask.astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (mask_blur,)*2, 0)/255
        mask = mask.reshape([h, w, 1])

        #cv2.imshow('image',mask)
        #cv2.waitKey(0)
        
        return mask

    def overlay_bg_blur(
            self, img0, img1,
            background_blur=51,
            mask_blur=201,
    ):

        blur1 = cv2.GaussianBlur(img1, (background_blur,)*2, 0)
        return combine_images(img1, blur1, self.blur_mask(img0))


def combine_images(img0, img1, alpha):
    dst = (1-alpha)*img1 + alpha*img0
    return np.clip(dst, 0, 255).astype(np.uint8)

###########################################################################


clf = FACE_FINDER()

for n in range(20,100):
    img_a = cv2.imread(f"samples/images_noise/level_{n:08d}_a.jpg")
    img_b = cv2.imread(f"samples/images_noise/level_{n:08d}_b.jpg")

    img = clf.overlay_bg_blur(img_a, img_b)

    cv2.imshow('image', img)
    cv2.waitKey(0)
