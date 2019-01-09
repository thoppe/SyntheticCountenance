import numpy as np
import cv2, imutils, dlib
from imutils import face_utils
from src.pipeline import Pipeline

print(f"dlib CUDA status: {dlib.DLIB_USE_CUDA}")
model_dest = 'model/dlib'
detector = dlib.get_frontal_face_detector()

'''
shape68_pred = dlib.shape_predictor(
    f'{model_dest}/shape_predictor_68_face_landmarks.dat')

shape5_pred = dlib.shape_predictor(
    f'{model_dest}/shape_predictor_5_face_landmarks.dat')

facerec = dlib.face_recognition_model_v1(
    f'{model_dest}/dlib_face_recognition_resnet_model_v1.dat')
'''


def compute(f_image, f_bbox, n_upsample=0):

    img = cv2.imread(f_image)
    faces = detector(img, n_upsample)

    if len(faces) != 1:
        print(f"REMOVING: {f_image}, {len(faces)} faces detected.")
        return os.remove(f_image)

    face = faces[0]
    bbox = [face.left(), face.top(), face.right(), face.bottom()]
    bbox = np.array(bbox)

    #print(f"Computed bbox {f_bbox}")
    np.save(f_bbox, bbox)

    '''
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
    '''


P = Pipeline(
    load_dest = 'samples/images',
    save_dest = 'samples/bbox',
    new_extension = 'npy',
)(compute)

