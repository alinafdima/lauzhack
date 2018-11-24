# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import os
import ipdb
from os.path import join

import numpy as np
import cv2
import scipy, scipy.signal
import pytesseract

from paths import data_path
from utils import *

def loadImage(filename):
    return cv2.imread(join(data_path, filename), cv2.IMREAD_GRAYSCALE)

def preprocessImg(img, type = 2):
    kernel  = np.ones((3,3),np.uint8)
    kernel[0,2] = kernel[2,2] = kernel[0,0] = kernel[2,0] = 0
    img2 = img.copy()

    # CHANGE!!!!
    img2 = img2[25:-25, 4:-25]

    if type == 1:
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    elif type == 2:
        img2 = scipy.signal.medfilt(img2, 3)
    return img2

def getConnComps(img):
    kernel3 = np.array([[0]*3, [1,1,1], [0]*3], np.uint8)
    img3 = img

    img3 = cv2.erode(img3, kernel3, iterations=50)
    img3 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel3.transpose(), iterations=5)
    ret, labels = cv2.connectedComponents(invert(img3))

    return img3, labels, ret

def getRange(labels, val):
    return cv2.boundingRect(np.uint8(labels==val))

def getSubImageByLabel(img, labels, val):
    x, y, w, h = getRange(labels, val)
    return img[y:y+h, x:x+w]

def padImage(img, padding=10):
    m,n = img.shape
    img2 = np.zeros((m+2*padding,n+2*padding))
    img2 = invert(img2)
    img2[padding:m+padding, padding:n+padding] = img
    return img2

# Uses the original image plus the result of connectedComponents
def coloredConnComps(img, labels, ret):
    # Map component labels to hue val
    label_hue = np.uint8(170*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # Apply to original image
    img4 = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)
    img4 = cv2.bitwise_or(img4, labeled_img)
    return img4

if __name__ == "__main__":
    img = loadImage('2017-01-20 - Lidl.png')
    img2 = preprocessImg(img)
    img3, labels, ret = getConnComps(img2)

    # cv2.imshow('1stcomp', getSubImageByLabel(img2, labels, 1))
    # cv2.imshow('hue', coloredConnComps(img2, labels, ret))

    subImg = getSubImageByLabel(img2, labels, 2)
    print(pytesseract.image_to_string(padImage(subImg, 20)))
