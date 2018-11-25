import PIL.Image
import IPython.display

import time
import re
import os
from os.path import join

import numpy as np
import cv2
import pytesseract

from paths import data_path


def invert(img):
    return np.uint8(255-img)

def getRange(labels, val):
    return cv2.boundingRect(np.uint8(labels==val))

def getSubImageByLabel(img, labels, val):
    x, y, w, h = getRange(labels, val)
    return img[y:y+h, x:x+w], (x,y,w,h)

def padImageTo(img, newSize):
    dx = newSize[0]-img.shape[0]
    px1 = px2 = py1 = py2 = 0

    if dx > 0:
        px1= int(dx/2)
        px2 = dx-px1
    
    dy = newSize[1]-img.shape[1]
    if dy > 0:
        py1 = int(dy/2)
        py2 = dy-py1
    
    return np.pad(img, ((px1, px2), (py1, py2)), 'constant', constant_values=255)

def padImage(img, padding=10):
    return np.pad(img, ((padding, padding), (padding, padding)), 'constant', constant_values=255)



def loadImage(filename):
    return cv2.imread(join(data_path, filename), cv2.IMREAD_GRAYSCALE)

def writeImage(filename, img):
    return cv2.imwrite(join(data_path, filename), img)

def readStoreLogos():
    D = {}
    for f in os.listdir(join(data_path, "output")):
        name = f.replace(".png", "")
        D[name] = loadImage("output/" + f)
    return D

def displayImage(img, label='Img', debug=False):
    cv2.imshow(label, img)
    if not debug:
        cv2.waitKey(0)

def imageToText(subImg, padding = 50):
    return pytesseract.image_to_string(padImage(subImg, padding))




def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in xrange(size_x):
        matrix [x, 0] = x
    for y in xrange(size_y):
        matrix [0, y] = y

    for x in xrange(1, size_x):
        for y in xrange(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1 )
            else:
                matrix [x,y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 1, matrix[x,y-1] + 1 )
    return 1-(1.0*matrix[size_x - 1, size_y - 1]) / (size_x + size_y)

def to_number(s, dft = float('nan')):
    s = str(filter(lambda ch: ch in "0123456789+-.,", s))
    if s:
        return float(s)
    return dft

def is_number(s):
    """ Returns True is string is a number. """
    return str(filter(lambda ch: ch not in ".,-+ ", s)).isdigit()


try:
    from cStringIO import StringIO
    def showarray(a, fmt='png'):
        a = np.uint8(a)
        f = StringIO()
        PIL.Image.fromarray(a).save(f, fmt)
        IPython.display.display(IPython.display.Image(data=f.getvalue()))
except:
    def showarray(a, fmt = ""):
        pass




__timer = None
def markTime():
    global __timer
    if __timer is None:
        __timer = time.time()
        return
    else:
        new_timer = time.time()
        print("--- %.3f seconds ---" % (time.time() - __timer))
        __timer = new_timer