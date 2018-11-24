# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import os, sys
import ipdb

import numpy as np
import cv2
import scipy, scipy.signal
import pytesseract

from utils import *
import stores

from receipt_object import Receipt


def fill_corners(img, padding=5l):
    img2 = invert(img)

    kernel = np.ones((3,3), np.uint8)
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel, iterations=5)

    ret, labels = cv2.connectedComponents(img2)

    markers = np.zeros(img.shape)

    tri_ur= np.tril(np.ones((padding, padding))).transpose()
    tri_ll= np.tril(np.ones((padding, padding)))
    tri_ul= np.fliplr(np.tril(np.ones((padding, padding))).transpose())
    tri_lr= np.fliplr(np.tril(np.ones((padding, padding))))

    markers[:padding,:padding] = tri_ul
    markers[:padding,-padding:] = tri_ur
    markers[-padding:,:padding] = tri_ll
    markers[-padding:,-padding:] = tri_lr
    markers = (1-img/255)*markers

    marked_labels = np.unique(labels*markers)

    # displayImage(coloredConnComps(img, labels, ret))

    img_out = np.copy(img)
    for label in marked_labels:
        if label != 0:
            img_out[labels==label] = 255

    return img_out


def preprocessImg(img, type = 2):
    kernel  = np.ones((3,3),np.uint8)
    kernel[0,2] = kernel[2,2] = kernel[0,0] = kernel[2,0] = 0
    img2 = img.copy()

    if type == 1:
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    elif type == 2:
        img2 = scipy.signal.medfilt(img2, 3)

    img2 = fill_corners(img2)
    
    return img2

# def getConnComps(img, iterationsErode=50):
#     kernel3 = np.array([[0]*3, [1,1,1], [0]*3], np.uint8)
#     kernel3t = kernel3.copy().transpose()
#     img3 = img

#     img3 = cv2.erode(img3, kernel3, iterations=iterationsErode)
#     img3 = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel3t, iterations=5)
#     # img3 = cv2.dilate(img3, kernel3, iterations=iterationsErode-10)
#     ret, labels = cv2.connectedComponents(invert(img3))

#     return img3, labels, ret

def compute_connected_components(receipt, iterationsErode=50):
    kernel = np.array([[0]*3, [1,1,1], [0]*3], np.uint8)
    kernel_t = kernel.copy().transpose()
    img = receipt.img.copy()

    img = cv2.erode(img, kernel, iterations=iterationsErode)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_t, iterations=5)
    ret, labels = cv2.connectedComponents(invert(img))

    receipt.conn_comp_labels = labels
    receipt.conn_comp_num = ret


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
    sy, sx = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])

    y1,x1 = img1.shape[0], img1.shape[1]
    img1 = cv2.resize(img1, (sy, int(1.0*x1*sy/y1)))

    y2,x2 = img2.shape[0], img2.shape[1]
    img2 = cv2.resize(img2, (sy, int(1.0*x2*sy/y2)))

    sx, sy = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])

    img1 = invert(padImageTo(img1, (sx,sy)))
    img2 = invert(padImageTo(img2, (sx,sy)))
    
    intersect = np.sum(cv2.bitwise_and(img1, img2))
    union = np.sum(cv2.bitwise_or(img1, img2))
    
    return 1.0*intersect / union


# --- Stores ---

def addToDict(D, logo):
    showarray(logo)
    name = raw_input("Enter the name of the Store: ")
    D[name] = logo
    writeImage("output/" + name + ".png", logo)
    return name

def detectStore(D, logoSubImg, threshold = 0.5):
    if logoSubImg.shape[0] > 200:
        return "<invalid>"
    showarray(logoSubImg)
    for k,v in D.items():
        print k, compImgs(logoSubImg, v)
        if compImgs(logoSubImg, v) > threshold:
            return k

    # return addToDict(D, logoSubImg)
    return "<unknown>"

def stripWhiteColumns(img, pos):
    x,y,w,h = pos

    x1 = 0
    for i in xrange(img.shape[1]):
        if sum(255-img[:, i])/255.0 > 1:
            x1 = i+1
            break

    x2 = img.shape[1]
    for i in xrange(img.shape[1]-1, x1, -1):
        if sum(255-img[:, i])/255.0 > 1:
            x2 = i
            break

    img = img[:, x1:x2]
    x = x+x1
    w = x2-x1

    return img, (x, y, w, h)

def compute_image_patches(receipt, threshLines = 10):
    img, labels, ret = receipt.img, receipt.conn_comp_labels, receipt.conn_comp_num
    L=[0] * (ret-1)

    for i in xrange(1,ret):
        L[i-1] = getSubImageByLabel(img, labels, i)
        L[i-1] = stripWhiteColumns(L[i-1][0], L[i-1][1])

    L = list(l for l in L if l[1][2] > 30)

    for i in xrange(len(L)):
        for j in xrange(i+1,len(L)):
            if L[j][1][1] < L[i][1][1] + threshLines:
                if L[i][1][0] > L[j][1][0]:
                    L[i], L[j] = L[j], L[i]
            else:
                break

    return L

def hackyGetLogo(filename):
    img = loadImage(filename)
    img2 = preprocessImg(img)
    img3, labels, ret = getConnComps(img2)
    logo, _ = getSubImageByLabel(img2, labels, 1)
    return logo

def fullStack(receipt, D):
    raw_img = loadImage(receipt.filename)
    receipt.img = preprocessImg(raw_img)
    compute_connected_components(receipt)

    receipt.logo, _ = getSubImageByLabel(receipt.img, receipt.conn_comp_labels, 1)
    store = detectStore(D, receipt.logo)
    receipt.patches = compute_image_patches(receipt)
    if store == "Lidl":
        stores.parseLidl(receipt)
    elif store == "Karstadt":
        stores.parseKarstadt(receipt)
    else:
        return "", None

def goThroughFilesToCheckLogo(D):
    for f in os.listdir(data_path):
        if not os.path.isfile(os.path.join(data_path, f)):
            continue
        logo = hackyGetLogo(f)
        print ""
        print f
        print imageToText(logo)
        print detectStore(D, logo)

def test1():
    print detectStore(D, hackyGetLogo("lidl/2017-01-20 - Lidl.png"))
    print detectStore(D, hackyGetLogo("2017-06-13 - Lidl.png"))
    print detectStore(D, hackyGetLogo("2017-06-17 - Lidl.png"))
    print detectStore(D, hackyGetLogo("2017-05-23 - Karstadt b.png"))
    print detectStore(D, hackyGetLogo("2017-05-23 - Karstadt c.png"))
    print detectStore(D, hackyGetLogo("2017-06-24 - Karstadt - Pants.png"))




def ex1():
    markTime()
    img = loadImage('lidl/2017-01-20 - Lidl.png')
    markTime()
    img2 = preprocessImg(img)
    markTime()
    img3, labels, ret = getConnComps(img2)
    markTime()

    logo, _ = getSubImageByLabel(img2, labels, 1)
    cv2.imwrite('output-1stcomp.png', )
    cv2.imwrite('output-hue.png', coloredConnComps(img2, labels, ret))

    subImg,_ = getSubImageByLabel(img2, labels, 2)
    print(pytesseract.image_to_string(padImage(subImg, 20)))

def main():
    D = readStoreLogos()
    receipt = Receipt(sys.argv[1])
    fullStack(receipt, D)

def debug_alina():
    img = loadImage('lidl/2017-01-20 - Lidl.png')
    # img = loadImage('2017-05-11 - Primark.png')

    img2 = preprocessImg(img)
    displayImage(img2)

if __name__ == "__main__":
    # debug_alina()
    main()
