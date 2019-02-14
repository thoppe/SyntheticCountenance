import cv2
from tqdm import tqdm
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.erode(mask, kernel, iterations = 20)

        mask = mask.astype(bool)
        mask = 255*mask.astype(np.uint8)
        mask = cv2.GaussianBlur(mask, (mask_blur,)*2, 0)/255
        mask = mask.reshape([h, w, 1])

        #cv2.imshow('image',mask)
        #cv2.waitKey(0)
        #exit()
        
        return mask

    def overlay_bg_blur(
            self, img0, img1,
            background_blur=51,
            mask_blur=201,
    ):

        blur1 = cv2.GaussianBlur(img1, (background_blur,)*2, 0)
        soft_mask = self.blur_mask(img0)
        dst = combine_images(img1, blur1, soft_mask)
        return dst, soft_mask


def combine_images(img0, img1, alpha):
    dst = (1-alpha)*img1 + alpha*img0
    return np.clip(dst, 0, 255).astype(np.uint8)

###########################################################################

clf = FACE_FINDER()

from bridson import poisson_disc_samples
width, height = 1024, 1024
width, height = 1920, 1080

pts = poisson_disc_samples(width=width, height=height, r=150)
pts = np.array(pts).astype(int)

start_idx = 28
IMG, MASK = [], []
for n in tqdm(range(len(pts))):
    
    img_a = cv2.imread(f"samples/images_noise/level_{n:08d}_a.jpg")
    img_b = cv2.imread(f"samples/images_noise/level_{n:08d}_b.jpg")

    try:
        img, soft_mask = clf.overlay_bg_blur(img_a, img_b)
    except:
        print("Failed an image")
        continue

    # Make the image larger if needed
    h, w = img.shape[:2]
    assert(height>=h)
    assert(width>=w)
    
    mean_pixel = img.mean(axis=0).mean(axis=0).astype(np.uint8)
    imgx = np.ones((height,width,3))*mean_pixel
    imgx[:h, :w] = img
    soft_maskx = np.zeros((height,width,3))
    soft_maskx[:h, :w] = soft_mask
    
    soft_mask = soft_maskx
    img = imgx

    #dx = np.random.randint(low=-512, high=512)
    #dy = np.random.randint(low=-512, high=512)
    dx,dy = pts[n]

    #angle = np.random.uniform(-10, 20)
    #M = cv2.getRotationMatrix2D((1024/2,1024/2),angle,1)
    #img = cv2.warpAffine(img,M,(1024,1024))
    #soft_mask = cv2.warpAffine(soft_mask,M,(1024,1024))

    img = np.roll(img, dx, axis=1)
    soft_mask = np.roll(soft_mask, dx, axis=1)
    
    img = np.roll(img, dy, axis=0)
    soft_mask = np.roll(soft_mask, dy, axis=0)

    IMG.append(img)
    MASK.append(soft_mask)


MASK = np.array(MASK)
img = np.zeros_like(IMG[0])

for src, mask in zip(IMG, MASK):
    #mx = (mask/norm)[:,:,np.newaxis]
    mx = mask

    print(img.shape, src.shape, mask.shape)
    img = combine_images(src, img, mask)
    
#print(img.shape)
#img = combine_images(IMG[0], IMG[1], MASK[0])
#img = combine_images(IMG[1], IMG[0], MASK[1])

cv2.imshow('image', img)
cv2.waitKey(0)

print(MASK.shape)
