#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os
import os.path as osp
from random import shuffle
from util import label_to_num, label_to_num_dom, label_to_num_com

LANDMAEK_NUM = 68

def parse_id_label(ground_truth_fp='../faces_224/anns/training_validation_new.txt'):
    # parse {id: label}
    lines = open(ground_truth_fp).read().replace('\r\n', '\n').strip().split('\n')
    id_label = {}

    for l in lines:
        #print(l)
        id, label = l.split()
        
        #id_label.update({id: label_to_num(label)})
        id_label.update({id: label})
        id_f = id[:-4] + '_f' + id[-4:]
        id_label.update({id_f: label})
        
    return id_label

def load_train_data_mean(data_dir='./faces_224/imgs', landmark_fp='../faces_224/anns/landmark_224.json', ground_truth_fp='../faces_224/anns/training_new.txt',norm_flag=False):
    id_label = parse_id_label(ground_truth_fp=ground_truth_fp)

    id_landmark = json.load(open(landmark_fp))

    # parse face_id_landmarks
    face_id_landmarks = {}
    for id, _ in id_label.items():
        face_id = id.split('_')[1]
        face_id_landmarks.update({face_id: []})

    for id, landmark in id_landmark.items():
        if id not in id_label.keys():
            continue
        face_id = id.split('_')[1]
        face_id_landmarks[face_id].append(landmark)

    face_id_landmarks_mean = {}
    for face_id, landmarks in face_id_landmarks.items():
        mean = np.zeros(LANDMAEK_NUM * 2, dtype=np.float32)
        cnt = 0
        for landmark in landmarks:
            mean += np.array(landmark, dtype=np.float32)
            cnt += 1
        mean /= cnt
        face_id_landmarks_mean.update({face_id: mean.flatten()})

    # parse mean landmark
    X, y = [], []
    for id, label in sorted(id_label.items()):  # if add sorted?
        face_id = id.split('_')[1]
        x = np.array(id_landmark[id], dtype=np.float32).flatten() - face_id_landmarks_mean[face_id]
        if norm_flag:
            x = (x - x.min()) / (x.max() - x.min())
        X.append(x)
        y.append(label)

    fps = [osp.join(data_dir, id.split('_')[1], id) for id, label in sorted(id_label.items())]
  #  y = np.array(y, dtype=np.uint32)
    return fps, X, y

def load_val_data_mean_f(data_dir='./faces_224/imgs', landmark_fp='../faces_224/anns/landmark_224.json', val_data_order='../faces_224/anns/order_of_validation.txt',norm_flag=False):
    lines = open(val_data_order).read().replace('\r\n', '\n').strip().split('\n')
    ids = []
    ids_f = []
    for l in lines:
        id = l.strip()
        ids.append(id)
        id_f = id[:-4] + '_f'+id[-4:]
        ids_f.append(id_f)
        
    # print(ids)
    id_landmark = json.load(open(landmark_fp))

    face_id_landmarks = {}
    for id in ids:
        face_id = id.split('_')[1]
        face_id_landmarks.update({face_id: []})

    # print(face_id_landmarks)
    for id in ids:
        face_id = id.split('_')[1]
        landmark = id_landmark[id]
        face_id_landmarks[face_id].append(landmark)

    for id in ids_f:
        face_id = id.split('_')[1]
        landmark = id_landmark[id]
        face_id_landmarks[face_id].append(landmark)

    face_id_landmarks_mean = {}
    for face_id, landmarks in face_id_landmarks.items():
        mean = np.zeros(LANDMAEK_NUM * 2, dtype=np.float32)
        cnt = 0
        for landmark in landmarks:
            mean += np.array(landmark, dtype=np.float32)
            cnt += 1
        mean /= cnt
        face_id_landmarks_mean.update({face_id: mean.flatten()})

    # parse mean landmark
    X = []
    for id in sorted(ids):  # if add sorted?
        face_id = id.split('_')[1]
        x = np.array(id_landmark[id], dtype=np.float32).flatten() - face_id_landmarks_mean[face_id]
        if norm_flag:
            x = (x - x.min()) / (x.max() - x.min())  # pay attention to this
        X.append(x)
    fps = [osp.join(data_dir, id.split('_')[1], id) for id in sorted(ids)]
    return fps, X

def load_val_data_mean(data_dir='./faces_224/imgs', landmark_fp='../faces_224/anns/landmark_224.json', val_data_order='../faces_224/anns/order_of_validation.txt',norm_flag=False):
    lines = open(val_data_order).read().replace('\r\n', '\n').strip().split('\n')
    ids = []
    for l in lines:
        id = l.strip()
        ids.append(id)

    # print(ids)
    id_landmark = json.load(open(landmark_fp))

    face_id_landmarks = {}
    for id in ids:
        face_id = id.split('_')[1]
        face_id_landmarks.update({face_id: []})

    # print(face_id_landmarks)
    for id in ids:
        face_id = id.split('_')[1]
        landmark = id_landmark[id]
        face_id_landmarks[face_id].append(landmark)

    face_id_landmarks_mean = {}
    for face_id, landmarks in face_id_landmarks.items():
        mean = np.zeros(LANDMAEK_NUM * 2, dtype=np.float32)
        cnt = 0
        for landmark in landmarks:
            mean += np.array(landmark, dtype=np.float32)
            cnt += 1
        mean /= cnt
        face_id_landmarks_mean.update({face_id: mean.flatten()})

    # parse mean landmark
    X = []
    for id in sorted(ids):  # if add sorted?
        face_id = id.split('_')[1]
        x = np.array(id_landmark[id], dtype=np.float32).flatten() - face_id_landmarks_mean[face_id]
        if norm_flag:
            x = (x - x.min()) / (x.max() - x.min())  # pay attention to this
        X.append(x)
    fps = [osp.join(data_dir, id.split('_')[1], id) for id in sorted(ids)]
    return fps, X

def create_list_train():
    ground_truth_fp = '../faces_224/anns/training_validation_new.txt'
    train_image_fps, train_landmarks, train_labels = load_train_data_mean(ground_truth_fp=ground_truth_fp,
                                                                          landmark_fp='../faces_224/anns/landmark_224.json',
                                                                          norm_flag=False)

    data = zip(train_image_fps, train_landmarks, train_labels)
    data = list(data)
    train_fp = '../faces_224/anns/train_ld_shuffle.txt'

    def write(wfp, data):
        records = []
        for fp, lm, lb in data:
            l = ''
            l += '{} '.format(fp)
            l += '{} '.format(lb)
            l += ' '.join(map(lambda s: '{:0.6f}'.format(s), lm.tolist()))
            records.append(l)
        shuffle(records)
        with open(wfp, 'w') as f:
            f.write('\n'.join(records))

    write(train_fp, data)

def create_list_psdo():
    ground_truth_fp = '../faces_224/anns/test_psdo.txt'
    train_image_fps, train_landmarks, train_labels = load_train_data_mean(ground_truth_fp=ground_truth_fp,
                                                                          landmark_fp='../faces_224/anns/landmark_224.json',
                                                                          norm_flag=False)

    data = zip(train_image_fps, train_landmarks, train_labels)
    data = list(data)
    train_fp = '../faces_224/anns/train_ld_psdo.txt'

    def write(wfp, data):
        records = []
        for fp, lm, lb in data:
            l = ''
            l += '{} '.format(fp)
            l += '{} '.format(lb)
            l += ' '.join(map(lambda s: '{:0.6f}'.format(s), lm.tolist()))
            records.append(l)
        shuffle(records)
        with open(wfp, 'w') as f:
            f.write('\n'.join(records))

    write(train_fp, data)

def create_list_test():
    #test_image_fps, test_landmarks = load_val_data_mean(val_data_order='../data/order_of_test.txt',
    #                                                    landmark_fp='../data/landmark_224.json',
    #                                                    norm_flag=False)
    val_image_fps, val_landmarks = load_val_data_mean_f(val_data_order= '../faces_224/anns/test_list.txt',
    landmark_fp='../faces_224/anns/landmark_224.json')
   # data = zip(test_image_fps, test_landmarks)
    data = zip(val_image_fps, val_landmarks)
    fp = '../faces_224/anns/test_ld.txt'

    def write(wfp, data):
        records = []
        for fp, lm in data:
            l = ''
            l += '{} '.format(fp)
            l += ' '.join(map(lambda s: '{:0.6f}'.format(s), lm.tolist()))
            records.append(l)
        with open(wfp, 'w') as f:
            f.write('\n'.join(records))
            
    write(fp, data)

if __name__ == '__main__':
    create_list_train()
    create_list_test()
   # create_list_psdo()
