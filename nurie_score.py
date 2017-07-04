# coding:utf-8

import sys
import cv2
import numpy as np

def distance_hsv(img1, img2):
    weight = 1.0 # 距離の重み
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

    # print(img1_hsv)

    tmp = (img1_hsv - img2_hsv).astype(float)
    h = tmp[:,:,0]
    s = tmp[:,:,1]
    v = tmp[:,:,2]
    s1s2 = (img1_hsv * img2_hsv).astype(float)[:,:,2]

    return np.sum(np.sqrt(weight * v * v + s * s + 2 * s1s2 * (1 - np.cos(h))))


def nurie_answer(answer_file, boundary_file):
    img_ans = cv2.imread(answer_file, 1)
    img_bou = cv2.imread(boundary_file, 1)

    # cv2.imwrite("./data/kamome_PD_answer2.png", np.minimum(img_ans, img_bou))

def nurie_score(test_file, orginal_file, best_test_file):
    img_test = cv2.imread(test_file, 1)
    img_org = cv2.imread(orginal_file, 1)
    img_best= cv2.imread(best_test_file, 1)

    width, height, ch = img_test.shape

    # print(img_test-img_ref)
    # pgbのユークリッド距離
    # tmp = (img_test-img_org).astype(float)
    # test_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    # print(test_score)
    # tmp = (img_best-img_org).astype(float)
    # best_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    # print(best_score)
    # tmp = np.maximum(img_test, 255-img_test).astype(float)
    # worst_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    # print(worst_score)

    # hsvによるいい感じの距離を考えたい。
    # hの違いは大きくスコアに乗るようにしたい

    test_score = distance_hsv(img_test, img_org)
    best_score = distance_hsv(img_best, img_org)


    # cv2.imshow("figure",img_test_hsv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # best_score = 0
    worst_score = 200000000
    return 100 - (test_score - best_score) / ( worst_score- best_score) * 100

if __name__ == "__main__":
    print(nurie_score("./data/output/paintbad.png", "./data/output/kamome_PD_answer1.png",  "./data/output/kamome_PD_answer2.png"))
    # nurie_answer("", "./data/output/kamome_PD_sp32_sr16_n8_it10_quest.png")
