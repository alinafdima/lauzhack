# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import os
import ipdb
from os.path import join

import numpy as np
import cv2

from paths import data_path


if __name__ == "__main__":
    img = cv2.imread(join(data_path, '2017-01-20 - Lidl.png'), cv2.IMREAD_GRAYSCALE)

    kernel  = np.ones((3,3),np.uint8)
    kernel[0,2] = kernel[2,2] = kernel[0,0] = kernel[2,0] = 0
    img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img2 = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel)
    # img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original',img)
    cv2.imshow('Processed',img2)
    # cv2.imwrite(filename + ".jpg", img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    print img
    print 'Eureka!'