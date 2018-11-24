import PIL.Image
import IPython.display

import time
from os.path import join

import numpy as np
import cv2

from paths import data_path


def invert(img):
    return np.uint8(255-img)

def getRange(labels, val):
    return cv2.boundingRect(np.uint8(labels==val))

def getSubImageByLabel(img, labels, val):
    x, y, w, h = getRange(labels, val)
    return img[y:y+h, x:x+w]

def padImageTo(img, newSize):
    dx = newSize[0]-img.shape[0]
    px1 = int(dx/2)
    px2 = dx-px1
    
    dy = newSize[1]-img.shape[1]
    py1 = int(dy/2)
    py2 = dy-py1
    
    return np.pad(img, ((px1, px2), (py1, py2)), 'constant', constant_values=255)

def padImage(img, padding=10):
    return np.pad(img, ((padding, padding), (padding, padding)), 'constant', constant_values=255)



def loadImage(filename):
    return cv2.imread(join(data_path, filename), cv2.IMREAD_GRAYSCALE)

def writeImage(filename, img):
    return cv2.imwrite(join(data_path, filename), img)

def displayImage(img, label='Img', debug=False):
    cv2.imshow(label, img)
    if not debug:
        cv2.waitKey(0)

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