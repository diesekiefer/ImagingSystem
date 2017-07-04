# coding:utf-8

import sys
import cv2
import numpy as np


def list_equal(a, b, n):
    for i in range(n):
        if a[i] != b[i]:
            return 1
    return 0

#　加工した画像を入力に最終的な塗り絵画像を出力したい
def makenurie(filename):
    img_src = cv2.imread(filename, 1)
    print(img_src.shape)
    width, height, ch = img_src.shape
    img_dst = np.ones((width, height))*255

    size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))
    matrix = [[1, 0, 1],
                [0, 1, 0]]
    affine_matrix = np.float32(matrix)
    img_left = cv2.warpAffine(img_src, affine_matrix, size, flags=cv2.INTER_LINEAR)
    img_left[:,0,:] = img_src[:,0,:]

    matrix = [[1, 0, -1],
                [0, 1, 0]]
    affine_matrix = np.float32(matrix)
    img_right = cv2.warpAffine(img_src, affine_matrix, size, flags=cv2.INTER_LINEAR)
    img_right[:,height-1,:] = img_src[:,height-1,:]

    matrix = [[1, 0, 0],
                [0, 1, 1]]
    affine_matrix = np.float32(matrix)
    img_top = cv2.warpAffine(img_src, affine_matrix, size, flags=cv2.INTER_LINEAR)
    img_top[0,:,:] = img_src[0,:,:]
    matrix = [[1, 0, 0],
                [0, 1, -1]]
    affine_matrix = np.float32(matrix)
    img_bot = cv2.warpAffine(img_src, affine_matrix, size, flags=cv2.INTER_LINEAR)
    img_bot[width-1,:,:] = img_src[width-1,:,:]

    img_dst = (np.all((img_src == img_left), axis=2) * np.all((img_src == img_right), axis=2) * np.all((img_src == img_top), axis=2) * np.all((img_src == img_bot), axis=2)).astype(int)
    print(img_dst)
    # for x in range(width):
    #     for y in range(height):
    #         # 周囲の画素の色をみて、それを元に自分の位置が黒になるか白になるかを決める
    #         my_color = img_src[x,y]
    #         left_color = img_src[x-1,y]
    #         if(x != 0):
    #             if list_equal(my_color, left_color, 3):
    #                 img_dst[x, y] = 0
    #         if(x != width-1):
    #             right_color = img_src[x+1,y]
    #             if list_equal(my_color, right_color, 3):
    #                 img_dst[x, y] = 0
    #         if(y != 0):
    #             top_color = img_src[x,y-1]
    #             if list_equal(my_color, top_color, 3):
    #                 img_dst[x, y] = 0
    #         if(y != height-1):
    #             bot_color = img_src[x,y+1]
    #             if list_equal(my_color, bot_color, 3):
    #                 img_dst[x, y] = 0

    # ギザギザしすぎているので、ギザギザしてるとこのは消したい

    # neiborhood4 = np.array([[1, 1, 1],
    #                         [1, 1, 1],
    #                         [1, 1, 1]],
    #                         np.uint8)
    #
    #
    #
    #
    # img_dst = cv2.erode(img_dst,
    #                           neiborhood4,
    #                           iterations=2)
    # img_dst = cv2.dilate(img_dst,
    #                           neiborhood4,
    #                           iterations=2)
    # # for x in range(width):
    #     for y in range(height):



    cv2.imshow("figure", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite("./data/hiyoko_nurie_morphology.png", img_dst)

if __name__ == "__main__":
    filename = "./data/kamome_PD.png"
    makenurie(filename)
