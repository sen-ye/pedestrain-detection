# num_train: pos: 2416 neg: 12180
# num_test: pos: 1126 neg: 4530
# train pos: 96*160
# test pos: 70*134
import os
import numpy as np
import cv2 as cv
import random
from PIL import Image


def set_seed(seed: int,):
    """ Set random seeds for numpy"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def pre(t):
    pos_dir = os.getcwd() + "\\data\\" + t + "\\pos\\"
    neg_dir = os.getcwd() + "\\data\\" + t + "\\neg\\"
    pos_list = os.listdir(pos_dir)
    neg_list = os.listdir(neg_dir)
    for i in pos_list:
        img = Image.open("./data/" + t + "/pos/" + i)
        img.save("./data_n/" + t + "/pos/" + i)
    for i in neg_list:
        img = Image.open("./data/" + t + "/neg/" + i)
        img.save("./data_n/" + t + "/neg/" + i)


def get_data(t):
    pre(t)
    pos_dir = os.getcwd() + "\\data_n\\" + t + "\\pos\\"
    neg_dir = os.getcwd() + "\\data_n\\" + t + "\\neg\\"
    pos_list = os.listdir(pos_dir)
    neg_list = os.listdir(neg_dir)
    hog_vec = []
    labels = np.zeros(len(pos_list)+10 * len(neg_list))
    labels[:len(pos_list)] = 1
    labels[len(pos_list):] = -1
    hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    for i in pos_list:
        img = cv.imread("./data_n/" + t + "/pos/" + i)
        if t == "train":
            img = img[16:144, 16:80]
        elif t == "test":
            img = img[3:131, 3:67]
        hog_com = hog.compute(img)
        hog_vec.append(hog_com)
    for i in neg_list:
        img = cv.imread("./data_n/" + t + "/neg/" + i)
        x, y, z = img.shape
        x -= 128
        y -= 64
        for k in range(10):
            xx = np.random.randint(0, x - 1)
            yy = np.random.randint(0, y - 1)
            hog_com = hog.compute(img[xx:xx+128, yy:yy+64])
            hog_vec.append(hog_com)
    hog_vec = np.float32(hog_vec)
    hog_vec = np.resize(hog_vec, (len(pos_list)+10 * len(neg_list), 3780))
    labels = np.int32(labels)
    return hog_vec, labels

