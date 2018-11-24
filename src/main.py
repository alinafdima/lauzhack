# -*- coding: utf-8 -*-
"""
@author: Alina Dima
"""

import os
from os.path import join
import cv2
from paths import data_path
import ipdb


if __name__ == "__main__":

    img = cv2.imread(join(data_path, '2017-01-20 - Lidl.png'), cv2.IMREAD_GRAYSCALE)
    print img
    print 'Eureka!'