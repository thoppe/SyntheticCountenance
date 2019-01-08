import os, json, glob
import numpy as np
from tqdm import tqdm
import random
import cv2, imutils, dlib
from imutils import face_utils

import dlib.cuda as cuda
cuda.set_device(0)

shape68_pred = dlib.shape_predictor(
    'keypoints_models/shape_predictor_68_face_landmarks.dat')

shape5_pred = dlib.shape_predictor(
    'keypoints_models/shape_predictor_5_face_landmarks.dat')

facerec = dlib.face_recognition_model_v1(
    'keypoints_models/dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()

def compute(f):

    f_json = f.replace('/imgs/', '/info/').replace('.jpg', '.json')

    if not os.path.exists(f_json):
        print(f"Removing {f}")
        os.remove(f)
        return False
    
    with open(f_json) as FIN:
        try:
            js = json.loads(FIN.read())
        except json.decoder.JSONDecodeError:
            print(f"Problem with json, removing {f}")
            os.remove(f)
            os.remove(f_json)
            return False

    #if 'faces' not in js:
    #    print(f"Missing faces in {f_json}, run P0_process_all_images.py")
    #    continue

    # Skip if there isn't a good single face
    #if len(js['faces']) != 1:
    #    #print("Not exactly one face in image")
    #    continue

    if 'dlib_faces' in js:
        return True
    
    img = cv2.imread(f)
    img = imutils.resize(img)
    faces = detector(img, 1)

    item = {}
    item['n'] = len(faces)

    face_data = []
    for face in faces:
        shape5 = shape5_pred(img, face)
        fd = facerec.compute_face_descriptor(img, shape5)
        fd = [float(x) for x in fd]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape68 = shape68_pred(gray, face)
        shape68x = face_utils.shape_to_np(shape68)
        keypoints = [list(map(int, (x,y))) for x,y in shape68x]

        face_data.append({
            'keypoints':keypoints,
            'face_vector':fd
        })

    item['faces'] = face_data
    js['dlib_faces'] = item

    json.dumps(js)
    
    with open(f_json, 'w') as FOUT:
        text = json.dumps(js)
        FOUT.write(text)
        
    print("Computed", f)



import joblib

F_JPG = sorted(glob.glob('../examples/imgs/*.jpg'))
random.shuffle(F_JPG)

ITR = tqdm(F_JPG)
for n in ITR:
    compute(n)

#with joblib.Parallel(1) as MP:
#    MP(joblib.delayed(compute(n) for n in ITR))

