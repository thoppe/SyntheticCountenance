import os, json, glob
from tqdm import tqdm
import random
import cv2, imutils, dlib
from imutils import face_utils

predictor = dlib.shape_predictor(
    'keypoints_models/shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()

F_JPG = glob.glob('../examples/imgs/*.jpg')
#random.seed(42)
random.shuffle(F_JPG)

for f in tqdm(F_JPG):

    f_json = f.replace('/imgs/', '/info/').replace('.jpg', '.json')

    if not os.path.exists(f_json):
        print(f"Removing {f}")
        os.remove(f)
        continue
    
    with open(f_json) as FIN:
        try:
            js = json.loads(FIN.read())
        except json.decoder.JSONDecodeError:
            print(f"Problem with json, removing {f}")
            os.remove(f)
            continue

    if 'faces' not in js:
        print("Run P0_process_all_images.py first")
        continue

    if len(js['faces']) != 1:
        #print("Not exactly one face in image")
        continue

    if 'keypoints' in js:
        continue

    face = js['faces'][0]['bounding_box']

    rect = dlib.rectangle(*face)

    img = cv2.imread(f)
    img = imutils.resize(img, width=500)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray)
    #rect = (face[0], face[1]), (face[2], face[3])

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    keypoints = [list(map(int, (x,y))) for x,y in shape]
    js['keypoints'] = keypoints
    
    with open(f_json, 'w') as FOUT:
        text = json.dumps(js)
        FOUT.write(text)
        
    print("Computed", f)
