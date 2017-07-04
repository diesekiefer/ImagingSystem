import sys
import cv2
import numpy as np
import time

class Nurie:
    def __init__(self, filepath="./data/hiyoko_PD.jpg"):
        self.filepath = filepath
        self.src = cv2.imread(filepath, 1)
        self.answer = None

    def makequestion(self, erode=False):
        """
        MAKEQUESTION
        :return: question image
        """
        # print(img_src)
        if self.answer is None:
            src = self.makeanswer()
        else:
            src = self.answer
        width, height, ch = src.shape
        img_dst = np.ones((width, height)) * 255

        size = tuple(np.array([src.shape[1], src.shape[0]]))
        matrix = [[1, 0, 1],
                    [0, 1, 0]]
        affine_matrix = np.float32(matrix)
        img_left = cv2.warpAffine(src, affine_matrix, size, flags=cv2.INTER_LINEAR)
        img_left[:,0,:] = src[:,0,:]

        matrix = [[1, 0, -1],
                    [0, 1, 0]]
        affine_matrix = np.float32(matrix)
        img_right = cv2.warpAffine(src, affine_matrix, size, flags=cv2.INTER_LINEAR)
        img_right[:,height-1,:] = src[:,height-1,:]

        matrix = [[1, 0, 0],
                    [0, 1, 1]]
        affine_matrix = np.float32(matrix)
        img_top = cv2.warpAffine(src, affine_matrix, size, flags=cv2.INTER_LINEAR)
        img_top[0,:,:] = src[0,:,:]
        matrix = [[1, 0, 0],
                    [0, 1, -1]]
        affine_matrix = np.float32(matrix)
        img_bot = cv2.warpAffine(src, affine_matrix, size, flags=cv2.INTER_LINEAR)
        img_bot[width-1,:,:] = src[width-1,:,:]

        img_dst = (np.all((src == img_left), axis=2) * np.all((src == img_right), axis=2) * np.all((src == img_top), axis=2) * np.all((src == img_bot), axis=2)).astype(np.uint8) * 255
        # for x in range(width):
        #     for y in range(height):
        #         # 周囲の画素の色をみて、それを元に自分の位置が黒になるか白になるかを決める
        #         my_color = src[x, y]
        #         left_color = src[x - 1, y]
        #         if x != 0:
        #             if self.list_equal(my_color, left_color, 3):
        #                 img_dst[x, y] = 0
        #         if x != width - 1:
        #             right_color = src[x + 1, y]
        #             if self.list_equal(my_color, right_color, 3):
        #                 img_dst[x, y] = 0
        #         if y != 0:
        #             top_color = src[x, y - 1]
        #             if self.list_equal(my_color, top_color, 3):
        #                 img_dst[x, y] = 0
        #         if y != height - 1:
        #             bot_color = src[x, y + 1]
        #             if self.list_equal(my_color, bot_color, 3):
        #                 img_dst[x, y] = 0

        if erode:
            img_dst = self.erodedilate(img_dst)
        return img_dst

    def makeanswer(self, sp=32, sr=32, n_cluster=16, it=10):
        """
        MAKEANSWER
        :param sp: 空間窓半径
        :param sr: 色空間窓半径
        :param n_cluster: 色の種類の最大値
        :param it: k-means の繰り返し回数
        :return: 答え
        """
        dst = cv2.pyrMeanShiftFiltering(self.src, sp=sp, sr=sr)
        print("Mean shift done.")
        dst = self.kmeanclustering(dst, n_cluster, it)
        self.answer = dst
        return dst

    @staticmethod
    def kmeanclustering(src, n_cluster=16, it=10):
        src_rsh = src.reshape((-1, 3))

        # float32に変換
        src_rsh = np.float32(src_rsh)

        # K-Means法
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(src_rsh,
                                        n_cluster,
                                        None,
                                        criteria,
                                        it,
                                        cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        print("K-means clustering done.")
        return res.reshape((src.shape))

    def list_equal(self, a, b, n):
        for i in range(n):
            if a[i] != b[i]:
                return 1
        return 0

    def erodedilate(self, img):
        neiborhood4 = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]],
                               np.uint8)

        img_dst = cv2.erode(img,
                            neiborhood4,
                            iterations=2)
        img_dst = cv2.dilate(img_dst,
                             neiborhood4,
                             iterations=2)
        return img_dst


if __name__ == "__main__":
    picturename = "kamome_PD"
    nr = Nurie("./data/{}.jpg".format(picturename))
    cv2.imshow("Source", nr.src)
    # cv2.waitKey(10)
    start = time.time()
    sp, sr, n, it = 32, 16, 8, 10
    img = nr.makeanswer(sp, sr, n, it)
    cv2.imshow("Answer",img)
    # cv2.waitKey(0)
    cv2.imwrite("./data/output/{}_sp{}_sr{}_n{}_it{}_ans.png".format(picturename, sp, sr, n, it), img)
    tmp = time.time()
    print ("answer_time:{0}".format(tmp - start) + "[sec]")
    quest = nr.makequestion(False)
    cv2.imshow("Question", quest)
    # cv2.waitKey(0)
    cv2.imwrite("./data/output/{}_sp{}_sr{}_n{}_it{}_quest.png".format(picturename, sp, sr, n, it), quest)

    end = time.time()
    print ("quest_time:{0}".format(end - tmp) + "[sec]")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
