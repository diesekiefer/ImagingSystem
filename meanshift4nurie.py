# coding:utf-8

import sys
import cv2
import numpy as np


# 位置と色でクラスタリングしてぺろっと貼ってみるか
def kmeans_clustering(img_src):
  # img_src = cv2.imread('./image/karasu.jpg')
  Z = img_src.reshape((-1,3))

  # float32に変換
  Z = np.float32(Z)

  # K-Means法
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 32
  ret,label,center=cv2.kmeans(Z,
                              K,
                              None,
                              criteria,
                              10,
                              cv2.KMEANS_RANDOM_CENTERS)

  # UINT8に変換
  center = np.uint8(center)
  res = center[label.flatten()]
  return res.reshape((img_src.shape))

  # cv2.imshow('Quantization', img_dst)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # cv2.imwrite('./data/hiyoko_kmeans.png', img_dst)

def meanshift(filename):
    img = cv2.imread(filename, 1)
    dst = cv2.pyrMeanShiftFiltering(img, 64, 64, 4)
    # canny_img = cv2.Canny(dst, 50, 110)
    cv2.imwrite("./data/hiyoko_meanshift.png", dst)
    # dst_gry = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./data/hiyoko_16.png", kmeans_clustering(img))
    # cv2.imwrite("./data/hiyoko.16_meanshift.png", kmeans_clustering(dst))

    # cv2.imshow('image1', kmeans_clustering(dst))
    # cv2.imshow('image2', ~canny_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == "__main__":
    filename = "./data/hiyoko_PD.jpg"
    meanshift(filename)
