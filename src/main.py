# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import os
import ipdb

import numpy as np
import cv2
import scipy, scipy.signal
import pytesseract

from utils import *

def fill_corners(img):
    (size_y, size_x) = img.shape
    # ipdb.set_trace()
    # corners = np.array([0, 0, 0, size_x-1, size_y-1, 0, size_y-1, size_x-1]).reshape((4,2))
    padding = 5

    markers = np.zeros(img.shape)

    tri_ur= 1*np.tril(np.ones((padding, padding))).transpose()
    tri_ll= 2*np.tril(np.ones((padding, padding)))
    tri_ul= 3*np.fliplr(np.tril(np.ones((padding, padding))).transpose())
    tri_lr= 4*np.fliplr(np.tril(np.ones((padding, padding))))

    markers[:padding,:padding] = tri_ul
    markers[:padding,-padding:] = tri_ur
    markers[-padding:,:padding] = tri_ll
    markers[-padding:,-padding:] = tri_lr

    ipdb.set_trace()

    img_3ch = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)
    markers = cv2.cvtColor(np.uint8(markers), cv2.COLOR_GRAY2BGR)

    img2 = cv2.watershed(img_3ch, markers)

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

def hackyGetLogo(filename):
    img = loadImage(filename)
    img2 = preprocessImg(img)
    img3, labels, ret = getConnComps(img2)
    return getSubImageByLabel(img2, labels, 1)

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

def compImgs(img1, img2):
    sx, sy = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
    img1 = invert(padImageTo(img1, (sx,sy)))
    img2 = invert(padImageTo(img2, (sx,sy)))
    
    intersect = np.sum(cv2.bitwise_and(img1, img2))
    union = np.sum(cv2.bitwise_or(img1, img2))
    
    return 1.0*intersect / union

def readStoreLogos():
    pass

def addToDict(D, logo):
    showarray(logo)
    name = raw_input("Enter the name of the Store: ")
    D[name] = logo
    writeImage("output/" + name + ".png", logo)
    return name

def detectStore(D, logoSubImg, threshold = 0.5):
    for k,v in D.items():
        print k, compImgs(logoSubImg, v)
        if compImgs(logoSubImg, v) > threshold:
            return k

    return addToDict(D, logoSubImg)

def main():
    markTime()
    img = loadImage('lidl/2017-01-20 - Lidl.png')
    markTime()
    img2 = preprocessImg(img)
    markTime()
    img3, labels, ret = getConnComps(img2)
    markTime()

    cv2.imwrite('output-1stcomp.png', getSubImageByLabel(img2, labels, 1))
    cv2.imwrite('output-hue.png', coloredConnComps(img2, labels, ret))

    subImg = getSubImageByLabel(img2, labels, 2)
    print(pytesseract.image_to_string(padImage(subImg, 20)))

def debug_alina():
    img = loadImage('2017-01-20 - Lidl.png')
    img2 = preprocessImg(img)
    img2 = fill_corners(img2)

if __name__ == "__main__":
    debug_alina()
    # main()
