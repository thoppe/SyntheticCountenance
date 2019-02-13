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

    def __call__(self, img):
        bbox = self.compute_bbox(img)
        pts = self.compute_keypoints(img, bbox)
        hull = cv2.convexHull(pts).squeeze()

        #kernel = np.ones((5,5),np.uint8)
        #expand = cv2.dilate(hull, kernel, iterations = 5)
        #return expand
    
        return hull

clf = FACE_FINDER()

n = 13
for n in range(20,100):
    img_a = cv2.imread(f"samples/images_noise/level_{n:08d}_a.jpg")

    img = img_a
    hull = clf(img)
    mask = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
    mask = cv2.drawContours(mask, [hull], 0, 255, -1)
    mask = mask.astype(bool)

    mask_blur = 201
    background_blur = 51
    smooth_mask = 255*mask.astype(np.uint8)
    smooth_mask = cv2.GaussianBlur(smooth_mask, (mask_blur,)*2, 0)/255
    smooth_mask = smooth_mask.reshape([1024, 1024, 1])

    #cv2.imshow('image',smooth_mask)
    #cv2.waitKey(0)
    
    img = cv2.imread(f"samples/images_noise/level_{n:08d}_b.jpg")
    blur = cv2.GaussianBlur(img, (background_blur,)*2, 0)

    MX = (1-smooth_mask)*blur + smooth_mask*img
    MX = np.clip(MX, 0, 255).astype(np.uint8)
    img = MX

    cv2.imshow('image',MX)
    cv2.waitKey(0)
