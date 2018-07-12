import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import pi
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc  # pip install Pillow
import matplotlib.pylab as pylab
import sys
import cv2

pylab.rcParams['figure.figsize'] = (20.0, 7.0)


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


# matplotlib inline
pylab.rcParams['figure.figsize'] = (20.0, 7.0)


def loadImage(path):
    return cv2.imread(path, 0)


def getImageSize(img):
    return img.shape


# Do 8x8 DCT on image (in-place)
def dct_tran(img):
    imsize = getImageSize(img)
    dct = np.zeros(imsize)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i + 8), j:(j + 8)] = dct2(img[i:(i + 8), j:(j + 8)])
    return dct


pos = 128


# Extract a block from image
# plt.figure()
# a = plt.imshow(im[pos:pos + 8, pos:pos + 8], cmap='gray')
# plt.title("An 8x8 Image block")

# Display the dct of that block
# plt.figure()
# plt.imshow(dct[pos:pos + 8, pos:pos + 8], cmap='gray', vmax=np.max(dct) * 0.01, vmin=0, extent=[0, pi, pi, 0])
# plt.title("An 8x8 DCT block")

# plt.figure()
# plt.imshow(dct, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
# plt.title("8x8 DCTs of the image")
# res = dct[pos:pos + 8, pos:pos + 8]


# showImage(small)

# Threshold
def threshhold_res(dct, thresh):
    return dct * (abs(dct) > (thresh * np.max(dct)))


# f, axarr = plt.subplots(2, 2)
# plt.imshow(dct_thresh, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
# plt.title("Thresholded 8x8 DCTs of the image")

def calculate_percent_nonzeros(dct_thresh, img):
    img_size = getImageSize(img)
    return np.sum(dct_thresh != 0.0) / (img_size[0] * img_size[1] * 1.0)


# print "Keeping only %f%% of the DCT coefficients" % percent_nonzeros

def idct_trans(img, dct_thresh):
    img_size = getImageSize(img)
    im_dct = np.zeros(img_size)
    for i in r_[:img_size[0]:8]:
        for j in r_[:img_size[1]:8]:
            im_dct[i:(i + 8), j:(j + 8)] = idct2(dct_thresh[i:(i + 8), j:(j + 8)])
    return im_dct


def show_img_im_dct(img, im_dct, percent_nonzeros):
    plt.figure()
    plt.imshow(np.hstack((img, im_dct)), cmap='gray')
    plt.title("Comparison between original and DCT compressed images with %f%% of the cooeficents" % percent_nonzeros)
    plt.show()


if __name__ == '__main__':
    print ('start')
    img = loadImage('lena512.bmp')
    dct = dct_tran(img)
    dct_thresh = threshhold_res(dct, 0.1)
    percent_nonzeros = calculate_percent_nonzeros(dct_thresh, img)
    idct = idct_trans(img, dct_thresh)
    show_img_im_dct(img, idct, percent_nonzeros*100)
