from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import numpy as np

# inspired on https://gist.github.com/smex/5287589
def numpy_to_qimage(img):
    '''
    skimage considers the following convention for the array organization:
        2D grayscale (row/height, col/width)
        2D multichannel (row/heigth, col/width, channel) # RGB, for example

    QImage constructors considers:
    QImage(uchar *data, int width, int height, int bytesPerLine, Format
    format, ...)

    bytesPerLine is the number of bytes per line in the image, i.e.,
    width * number_bytes_per_pixel
    E.g: For a 8 bits (1 byte) grayscale image of width=400, height=300,
    we have bytesPerLine = 400 * 1 = 400

    For a RGB888 (8 bits (1 byte) per channel) image of width=400, height=300,
    we have bytesPerLine = 400 * 3 * 1 = 2400

    The numpy.ndarray.stride stores exactly such information:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html
    '''

    if img is None:
        return QImage()

    gray_color_table = [qRgb(i, i, i) for i in range(256)]

    if img.dtype == np.uint8:
        qimg = None

        # Gray Image
        if len(img.shape) == 2:
            qimg = QImage(img.data, img.shape[1], img.shape[0],
                          img.strides[0], QImage.Format_Indexed8)
            qimg.setColorTable(gray_color_table)

        # Color Image
        elif len(img.shape) == 3:
            if img.shape[2] == 3: # RGB
                qimg = QImage(img.data, img.shape[1], img.shape[0],
                              img.strides[0], QImage.Format_RGB888)
            elif img.shape[2] == 4:
                qimg = QImage(img.data, img.shape[1], img.shape[0],
                              img.strides[0], QImage.Format_ARGB32)

        return qimg