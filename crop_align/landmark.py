#!/usr/bin/env python
# coding: utf-8

from util import *
from glob import glob
import os.path as osp
import dlib
from multiprocessing import Pool
import json

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
face_detector = dlib.get_frontal_face_detector()
face_regressor = dlib.shape_predictor(PREDICTOR_PATH)

def get_bbox(img, scale=8):
    """resize the image by scale, to accelerate the face bbox detection, and do some
    inversion calculation to fit image origin size"""

    # resize
    h, w, c = img.shape
    dsize = (w // scale, h // scale)
    img_scale = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)

    rects = face_detector(img_scale, 1)
    if len(rects) == 0:
        # print('Not detected')
        h, w, _ = img.shape
        return 0, 0, w, h
    rect = rects[0]
    bbox = l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
    bbox = map(lambda x: scale * x, bbox)
    return bbox

def get_landmark(param):
    """a function for map, parse landmark"""
    fp, scale = param
    img = gjz_read_img(fp)

    result = {}

    bbox = get_bbox(img, scale=scale)
    rect = dlib.rectangle(*bbox)
    landmark = face_regressor(img, rect).parts()
    res = []
    for pt in landmark:
        res.append(pt.x)
        res.append(pt.y)

    id_landmark = osp.split(fp)[-1]
    
    result[id_landmark] = res
#####
    img_flip =  cv2.flip(img, 1)
    bbox_flip = get_bbox(img_flip, scale=scale)
    rect_flip = dlib.rectangle(*bbox_flip)
    landmark_flip = face_regressor(img_flip, rect_flip).parts()
    res_flip = []
    for pt in landmark_flip:
        res_flip.append(pt.x)
        res_flip.append(pt.y)

    id_landmark_flip = id_landmark[:-4] + '_f' + id_landmark[-4:]
    result[id_landmark_flip] = res_flip
   # print(id_landmark, id_landmark_flip)
    return result

def get_landmark_pool(data_dir, wfp, processes=24, scale=8):
    """Use multiple process to do face detection and landmark task, and write the result
    to json file"""
    pool = Pool(processes=processes)  # will use all the processor in computer

    fps = sorted(glob(osp.join(data_dir, '*.JPG')))
    scales = [scale] * len(fps)
    res = pool.map(get_landmark, zip(fps, scales))

    pool.close()
    pool.join()

    id_landmark = {}
    for r in res:
        id_landmark.update(r)
    # set sort_keys True to keep its default order
    json.dump(id_landmark, open(wfp, 'w'), sort_keys=True)

def main():
    get_landmark_pool(processes=28, data_dir='/data/zzg/Emotion_challenge/Training', wfp='../faces_224/anns/train_landmark.json')
    get_landmark_pool(processes=28, data_dir='/data/zzg/Emotion_challenge/Validation', wfp='../faces_224/anns/val_landmark.json')
    get_landmark_pool(processes=28, data_dir='/data/zzg/Emotion_challenge/Test', wfp='../faces_224/anns/test_landmark.json')
    #get_landmark_pool(data_dir='../data/test_data', wfp='../data/test_landmark.json')

if __name__ == '__main__':
    main()
