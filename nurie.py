import sys
import cv2
import numpy as np


class Nurie:
    def __init__(self, filepath="./data/hiyoko_PD.jpg"):
        self.filepath = filepath
        self.src = cv2.imread(filepath, 1)
        self.answer = None

    def makequestion(self):
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
        for x in range(width):
            for y in range(height):
                # 周囲の画素の色をみて、それを元に自分の位置が黒になるか白になるかを決める
                my_color = src[x, y]
                left_color = src[x - 1, y]
                if x != 0:
                    if self.list_equal(my_color, left_color, 3):
                        img_dst[x, y] = 0
                if x != width - 1:
                    right_color = src[x + 1, y]
                    if self.list_equal(my_color, right_color, 3):
                        img_dst[x, y] = 0
                if y != 0:
                    top_color = src[x, y - 1]
                    if self.list_equal(my_color, top_color, 3):
                        img_dst[x, y] = 0
                if y != height - 1:
                    bot_color = src[x, y + 1]
                    if self.list_equal(my_color, bot_color, 3):
                        img_dst[x, y] = 0
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


if __name__ == "__main__":
    nr = Nurie("./data/blue-tit.jpg")
    cv2.imshow("Source", nr.src)
    cv2.waitKey(0)
    img = nr.makeanswer(32,32,16,10)
    cv2.imshow("Answer",img)
    cv2.waitKey(0)
    quest = nr.makequestion()
    cv2.imshow("Question", quest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
