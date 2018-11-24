import PIL.Image
import IPython.display

import numpy as np

try:
    from cStringIO import StringIO
except:
    pass

def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

def invert(img):
    return np.uint8(255-img)