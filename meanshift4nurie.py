# coding:utf-8

import sys
import cv2
import numpy as np

def meanshift(filename):
    img = cv2.imread(filename, 1)
    dst = cv2.pyrMeanShiftFiltering(img, 30, 30)
    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    filename = "/Users/diesekiefer/work/lectures/imagingsystem/data/town1.jpg"
    meanshift(filename)
