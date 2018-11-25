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
    square= np.ones((padding, padding))

    midx, midy = (img.shape[0]-padding)/2, (img.shape[1]-padding)/2
    markers[:padding,:padding] = tri_ul
    markers[:padding,-padding:] = tri_ur
    markers[-padding:,:padding] = tri_ll
    markers[-padding:,-padding:] = tri_lr
    markers[midx:midx+padding, 0:padding] = square
    markers[midx:midx+padding, -padding:] = square
    markers[0:padding, midy:midy+padding] = square
    markers[-padding:, midy:midy+padding] = square
    markers = (1-img/255)*markers

    marked_labels = np.unique(labels*markers)

    # displayImage(coloredConnComps(img, labels, ret))

    img_out = np.copy(img)
    for label in marked_labels:
        if label != 0:
            img_out[labels==label] = 255

    return img_out


def preprocessImg(img, type = 2, return_intermediate=False):
    kernel  = np.ones((3,3),np.uint8)
    kernel[0,2] = kernel[2,2] = kernel[0,0] = kernel[2,0] = 0
    img2 = img.copy()

    if type == 1:
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    elif type == 2:
        img2 = scipy.signal.medfilt(img2, 3)

    img3 = fill_corners(img2)
    
    if return_intermediate:
        return (img2, img3)
    else:
        return img3


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

    img2 = stripImgWhite(img2)

    img1 = invert(cv2.resize(img1, (sx, sy)))
    img2 = invert(cv2.resize(img2, (sx, sy)))
    
    intersect = np.sum(cv2.bitwise_and(img1, img2))
    union = np.sum(cv2.bitwise_or(img1, img2))
    
    return 1.0*intersect / union


# --- Stores ---

def addToDict(D, logo, name = ""):
    showarray(logo)
    if not name:
        name = raw_input("Enter the name of the Store: ")
    else:
        print(name)
    D[name] = logo
    writeImage("output/" + name + ".png", logo)
    return name

def detectStore(D, logoSubImg, threshold = 0.55, verbose = False):
    if logoSubImg.shape[0] > 200:
        return "<invalid>"

    if verbose:
        showarray(logoSubImg)
        print "Comparison Scores"

    bestStore = "<unknown>"
    bestValue = threshold
    for k,img in D.items():
        val = compImgs(logoSubImg, img)
        if verbose:
            print "%20s %.6f"%(k, val)
        if val > bestValue:
            bestStore = k
            bestValue = val

    # return addToDict(D, logoSubImg)
    return bestStore

class ImagePatch:
    def __init__(self):
        self.img = None
        self.bbox  = None
        self.text  = None

    def getText(self):
        if self.text is None:
            self.text = imageToText(self.img)
        return self.text


def stripImgWhite(img):
    for i in xrange(img.shape[1]):
        if sum(255-img[:, i])/255.0 > 1:
            x1 = i+1
            break

    x2 = img.shape[1]
    for i in xrange(img.shape[1]-1, x1, -1):
        if sum(255-img[:, i])/255.0 > 1:
            x2 = i
            break
    return img[:, x1:x2]

def stripWhiteColumns(patch):
    x,y,w,h = patch.bbox
    img = patch.img

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

    patch.img = img[:, x1:x2]
    patch.bbox = (x+x1, y, x2-x1, h)

def compute_image_patches(receipt, threshLines = 10):
    img, labels, ret = receipt.img, receipt.conn_comp_labels, receipt.conn_comp_num
    L=[0] * (ret-1)

    for i in xrange(1,ret):
        L[i-1] = ImagePatch()
        L[i-1].img, L[i-1].bbox = getSubImageByLabel(img, labels, i)
        stripWhiteColumns(L[i-1])

    L = list(l for l in L if l.bbox[2] > 30)

    for i in xrange(len(L)):
        for j in xrange(i+1,len(L)):
            if L[j].bbox[1] < L[i].bbox[1] + threshLines:
                if L[i].bbox[0] > L[j].bbox[0]:
                    L[i], L[j] = L[j], L[i]
            else:
                break

    return L



def addAllLogos(D):
    D = readStoreLogos()
    L = {("2017-01-20 - Lidl.png", "Lidl"), \
         ("2017-05-11 - Primark.png", "Primark"), \
         ("2017-05-23 - Karstadt b.png", "Karstadt"), \
         ("2017-07-11 - Karstadt.png", "Karstadt Feinkost"), \
         ("2017-07-11 - Oishii.png", "Oishii"), \
         ("2017-07-13 - dm.png", "dm"), \
         ("2017-07-22 - Aldi.png", "Aldi Sued"), \
         ("2017-09-09 - TKMaxx.png", "TKMaxx"), \
         ("2017-09-16 - Galeria Kaufhof.png", "Galeria Kaufhof"), \
         ("2017-09-22 - Rewe.png", "Rewe"), \
         ("2017-10-01 - Rossman.png", "Rossman"), \
         ("2017-10-21 - Gamestop.png", "Gamestop US") }
         # "2017-09-02 - C&A.png": "CandA", \

    for img_file, name in L:
        print img_file,
        receipt = parseReceipt(img_file, D, verbose = False)
        print receipt.store
        if receipt.store =="<unknown>":
            addToDict(D, receipt.logo, name)

def goThroughFilesToCheckLogo(D):
    for f in os.listdir(data_path):
        if not os.path.isfile(os.path.join(data_path, f)):
            continue
        parseReceipt(f, D, verbose = True)

def test1():
    print detectStore(D, hackyGetLogo("lidl/2017-01-20 - Lidl.png"), verbose = True)
    print detectStore(D, hackyGetLogo("2017-06-13 - Lidl.png"), verbose = True)
    print detectStore(D, hackyGetLogo("2017-06-17 - Lidl.png"), verbose = True)
    print detectStore(D, hackyGetLogo("2017-05-23 - Karstadt b.png"), verbose = True)
    print detectStore(D, hackyGetLogo("2017-05-23 - Karstadt c.png"), verbose = True)
    print detectStore(D, hackyGetLogo("2017-06-24 - Karstadt - Pants.png"), verbose = True)

def test_all_lidl(D, parseItems = True):
    L = ["2017-01-20 - Lidl.png", "2017-05-16 - Lidl.png", "2017-06-13 - Lidl.png", \
        "2017-06-17 - Lidl.png", "2017-07-01 - Lidl.png", "2017-07-22 - Lidl.png", \
        "2017-08-26 - Lidl.png", "2017-09-16 - Lidl.png", "2017-12-04 - Lidl.png", \
        "2017-12-16 - Lidl.png", "2018-01-06 - Lidl.png", "2018-02-24 - Lidl.png", \
        "2018-03-03 - Lidl.png", "2018-03-10 - Lidl.png", "2018-05-05 - Lidl.png", \
        "2018-05-19 - Lidl.png", "2018-06-02 - Lidl.png"]

    for img_file in L:
        parseReceipt(img_file, D, verbose = True, parseItems = parseItems)

def test_item_parsing(D):
    L = ["2017-01-20 - Lidl.png", "2017-05-11 - Primark.png", "2017-05-23 - Karstadt.png", \
        "2017-06-13 - Lidl.png", "2017-07-11 - Karstadt.png", "2017-07-22 - Aldi.png"]

    for img_file in L:
        parseReceipt(img_file, D, verbose = True, parseItems = True)

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

    # img_file = 'lidl/2017-01-20 - Lidl.png'
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
        parseReceipt(img_file, D, verbose = True, parseItems = False)
    else:
        for img_file in os.listdir(data_path):
            if not os.path.isfile(os.path.join(data_path, img_file)):
                continue

            parseReceipt(img_file, D, verbose = False, parseItems = False)
            print '___________________________________________________________'



def parseReceipt(img_file, D, verbose = False, parseItems = False):
    if verbose:
        markTime()


    receipt = Receipt(img_file)
    raw_img = loadImage(receipt.filename)
    receipt.img = preprocessImg(raw_img)
    receipt.img_text = imageToText(receipt.img)
    compute_connected_components(receipt)
    receipt.patches = compute_image_patches(receipt)

    # displayImage(receipt.img)

    # receipt.logo, _ = getSubImageByLabel(receipt.img, receipt.conn_comp_labels, 1)
    receipt.logo = receipt.patches[0].img
    receipt.store = detectStore(D, receipt.logo, verbose = verbose)

    stores.parse_date(receipt)
    stores.parse_total(receipt)

    if parseItems:
        if receipt.store == "Lidl":
            stores.parseLidl(receipt)
        elif receipt.store in ("Karstadt Feinkost", "Aldi Sued", "Primark"):
            stores.parseKarstadt(receipt)


    if verbose:
        markTime()

        print "\n--- Receipt ---"
        print "Store:", receipt.store
        print "Date: ", receipt.date
        print "Paid: ", receipt.total

        if parseItems:
            print "\nItems:"
            for item in receipt.items:
                qty_str = ""
                if "qty" in item:
                    qty_str = "%s x %s"%(item["qty"], item["unitprice"])
                print "%50s %10s %s, VAT %s"%(item["title"], qty_str, item["price"], item["vat"])
    return receipt


if __name__ == "__main__":
    main()


