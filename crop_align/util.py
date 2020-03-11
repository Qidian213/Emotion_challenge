#!/usr/bin/env python
# coding: utf-8

import cv2
#import matplotlib.pyplot as plt


def gjz_read_img(filename='', flag=cv2.IMREAD_COLOR):
    # for dlib face landmark detection, must return uint8 type ndarray
    img = cv2.imread(filename, flag)
    # return img.astype(np.float32)
    return img


def gjz_write_img(filename=None, img=None):
    """A wrapper of `cv2.imwrite`, but considering some situations"""
    assert (filename is not None and img is not None)

    cv2.imwrite(filename, img)

def label_to_num(lb):
    if 'N' in lb:
        return 0
    com, dom = map(int, lb.split('_'))
    n = (com - 1) * 7 + dom
    return n

def num_to_label(n):
    if n == 0:
        return 'N_N'
    dom = (n - 1) % 7 + 1
    com = (n - 1) // 7 + 1
    return '{}_{}'.format(com, dom)


def label_to_num_com(lb):
    if 'N' in lb:
        return 0
    com, _ = map(int, lb.split('_'))
    return com


def label_to_num_dom(lb):
    if 'N' in lb:
        return 0
    _, dom = map(int, lb.split('_'))
    return dom