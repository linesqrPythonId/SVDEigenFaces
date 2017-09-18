import numpy as np
import math


def getClassnumber(class_label, unkomega, testing_vec, bobot):
    unknown = np.asarray(unkomega)

    jarak = euclideanDistance(unknown, bobot)

    nomor = int(np.floor(np.argmin(jarak) * len(testing_vec)) / len(bobot))

    return nomor


def euclideanDistance(unkomega, bobot):
    jarak = []
    for index in range(len(bobot)):
        bt = np.asarray(bobot[index])
        dist = 0
        for indexBobot in range(len(bt)):
            dist += math.pow(math.fabs(bt[indexBobot] - unkomega[indexBobot]), 2)
        dist = math.sqrt(dist)
        jarak.append(dist)
    return jarak
