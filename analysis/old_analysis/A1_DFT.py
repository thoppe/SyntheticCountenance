import cv2
import os, sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct

n = int(sys.argv[1])


def show(n):
    f_img = f"../examples/imgs/{n:06}.jpg"
    assert(os.path.exists(f_img))

    img = cv2.imread(f_img, 0)

    #dc = np.clip(dc, 1, dc.max())
    #dc = np.log(np.log(dc))

    #w = np.fft.fft2(img)
    #fshift = np.fft.fftshift(w)
    #magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(2,2,1)
    plt.imshow(img, cmap = 'gray')
    plt.title(f'Input Image {n}'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,2)

    dc = dct(img)
    y = np.clip(np.log(dc.sum(axis=0)**2), 0, 50)
    plt.plot(y,alpha=0.8)

    dc = dct(img.T)
    y = np.clip(np.log(dc.sum(axis=0)**2), 0, 50)
    plt.plot(y,alpha=0.8)

    plt.ylim(0,40)

    plt.subplot(2,2,3)
    edges = cv2.Canny(img,50,50, apertureSize=3)
    plt.imshow(edges, cmap = 'gray')

    plt.subplot(2,2,4)
    edges = cv2.Canny(img,100,100, apertureSize=3)
    plt.imshow(edges, cmap = 'gray')
        

    #plt.imshow(dct(img), cmap = 'gray')
    #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()


for n in range(7, 200):
    show(n)
