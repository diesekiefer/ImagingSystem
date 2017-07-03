# coding:utf-8

import sys
import cv2
import numpy as np

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
    tmp = (img_test-img_org).astype(float)
    test_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    print(test_score)
    tmp = (img_best-img_org).astype(float)
    best_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    print(best_score)
    tmp = np.maximum(img_test, 255-img_test).astype(float)
    worst_score = np.sum(np.sqrt(np.sum(tmp*tmp, axis=2)))
    print(worst_score)


    best_score = 0
    return 100 - (test_score - best_score) / (worst_score - best_score) * 100

if __name__ == "__main__":
    print(nurie_score("./data/output/kamome_PD_answer2.png", "./data/output/kamome_PD_answer1.png",  "./data/output/kamome_PD_answer2.png"))
    # nurie_answer("", "./data/output/kamome_PD_sp32_sr16_n8_it10_quest.png")
