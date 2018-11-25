# -*- coding: utf-8 -*-
"""
Functions used in the demo
"""

import scipy.misc
import numpy as np

def showLogoImages(D, height = 50):
    images = []
    tmp_images = []
    spacer = 255 * np.ones((height, height))
    # spacer[height/2 - 10: height/2 + 10, height/2 - 10: height/2 + 10] = 0

    for i, (k, img) in enumerate(D.items()):
        factor = height * 1.0 / img.shape[0]
        tmp_images.append(scipy.misc.imresize(img, factor))
        tmp_images.append(spacer)

        if i % 3 == 2:
            images.append(np.concatenate(tmp_images, axis=1))
            tmp_images = []

    return images