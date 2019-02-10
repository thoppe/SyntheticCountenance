import numpy as np
import cv2, imutils, dlib, os
from imutils import face_utils
import PIL.Image
import PIL.ImageFile
import scipy.ndimage


class Image_Transformer:

    """
    Gets the image in shape for the GAN (for inverse or training)
    """

    def __init__(self):
        self.is_loaded = False

    def _load_models(self):
        if self.is_loaded:
            return True

        model_dest = "model/dlib"
        self.shape68_pred = dlib.shape_predictor(
            f"{model_dest}/shape_predictor_68_face_landmarks.dat"
        )
        self.detector = dlib.get_frontal_face_detector()

        self.is_loaded = True

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

    def __call__(self, f_img):
        '''
        Applies the prep to get the image ready for styleGAN.
        Returns a PIL model.
        '''
        
        self._load_models()

        img = cv2.imread(f_img)
        bbox = self.compute_bbox(img)
        key68 = self.compute_keypoints(img, bbox)

        # Adapting from
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        lm = key68
        output_size = 1024
        transform_size = 4096
        enable_padding = True

        lm_chin = lm[0:17]  # left-right
        lm_eyebrow_left = lm[17:22]  # left-right
        lm_eyebrow_right = lm[22:27]  # left-right
        lm_nose = lm[27:31]  # top-down
        lm_nostrils = lm[31:36]  # top-down
        lm_eye_left = lm[36:42]  # left-clockwise
        lm_eye_right = lm[42:48]  # left-clockwise
        lm_mouth_outer = lm[48:60]  # left-clockwise
        lm_mouth_inner = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        img = PIL.Image.open(f_image)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (
                int(np.rint(float(img.size[0]) / shrink)),
                int(np.rint(float(img.size[1]) / shrink)),
            )
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]),
        )
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0),
        )
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(
                np.float32(img),
                ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                "reflect",
            )
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0
                - np.minimum(
                    np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]
                ),
                1.0
                - np.minimum(
                    np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]
                ),
            )
            blur = qsize * 0.02
            img += (
                scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
            ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(
                np.uint8(np.clip(np.rint(img), 0, 255)), "RGB"
            )
            quad += pad[:2]

        # Transform.
        img = img.transform(
            (transform_size, transform_size),
            PIL.Image.QUAD,
            (quad + 0.5).flatten(),
            PIL.Image.BILINEAR,
        )

        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        return img


    def get_mask_from_file(self, f_img):
        '''
        Returns the mask from an image file
        '''
        
        self._load_models()
        img = cv2.imread(f_img)

        img = cv2.imread(f_img)
        bbox = self.compute_bbox(img)
        keypoints = self.compute_keypoints(img, bbox)

        hull = cv2.convexHull(keypoints)
        mask = np.zeros(img.shape, np.uint8)

        cv2.drawContours(mask, [hull], 0, 1, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = mask.astype(bool)[:, :, 0]

        return mask
        
if __name__ == "__main__":
    clf = Image_Transformer()

    f_image = "../../../Desktop/maxresdefault.jpg"
    #f_image = "../../../Desktop/00240.jpg"

    img_out = clf(f_image)
    f_save = "demo_transform.jpg"

    # Save aligned image.
    img_out.save(f_save)

    mask = clf.get_mask_from_file(f_save)
    print(mask.mean())
