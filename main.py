from EigenFaces import util, subspace, evaluasi, WebcamHandler
import numpy as np
import cProfile
from time import time
from memory_profiler import profile
import json


# @profile
def mainAll(path="Training", size=250, output="EigenUtil"):
    if path == None:
        path = "Training"
    set_of_images, label = util.readImageFromPath(root_path=path, size=size)
    print("set_of_images", np.shape(set_of_images))

    average = util.computeAverage(set_of_images)
    print("average", np.shape(average))

    subtracAvg = util.computeSubtractedAverage(set_of_images, average)
    print("subtracAvg", np.shape(subtracAvg))
    print("subtracAvg", len(subtracAvg))

    matCov = util.computeCovarianceMatrix(subtracted_average=subtracAvg)
    print("matCov", np.shape(matCov))

    eVal, eVec = subspace.computeEigenSmall(covariance_matrix=matCov, subtracted_average=subtracAvg)

    print(np.shape(eVal))
    print(np.shape(np.asarray(eVec)))

    eVec = util.normalization(eVec)
    eigenface = util.setEigenface(25, eVec)

    bobot = util.getWeight(eigenFaces=eigenface, subtracted_average=subtracAvg)
    print("bobot: ", np.shape(bobot))
    print("bobot: ", len(bobot))

    util.saveTrainingToNPZ(eigenFaces=eigenface, subtracted_average=subtracAvg, average=average, label=label,
                           bobot=bobot, path=output)


@profile()
def mainCustom(path="Training", trainingNumber=4, size=250):
    set_of_images, label = util.readImageFromPathTraining(root_path=path, trainingNumber=trainingNumber, size=size)
    print("set_of_images", np.shape(set_of_images))

    average = util.computeAverage(set_of_images)
    print("average", np.shape(average))

    subtracAvg = util.computeSubtractedAverage(set_of_images, average)
    print("subtracAvg", np.shape(subtracAvg))
    print("subtracAvg", len(subtracAvg))
    print("subtracAvg", len(subtracAvg[0]))

    matCov = util.computeCovarianceMatrix(subtracted_average=subtracAvg)
    print("matCov", np.shape(matCov))

    eVal, eVec = subspace.computeEigenSmall(covariance_matrix=matCov, subtracted_average=subtracAvg)

    print(np.shape(eVal))
    print(np.shape(np.asarray(eVec)))

    eVec = util.centralization(eVec)
    eVec = util.normalization(eVec)
    eigenface = util.setEigenface(25, eVec)

    bobot = util.getWeight(eigenFaces=eigenface, subtracted_average=subtracAvg)
    print("bobot: ", np.shape(bobot))
    print("bobot: ", len(bobot))
    # util.saveEigenUtil(eigenFaces=eigenface, subtracted_average=subtracAvg, average=average, label=label, bobot=bobot,
    # number_of_training=trainingNumber, path="EigenUtil/BPPT")
    util.saveTrainingToNPZ(eigenFaces=eigenface, subtracted_average=subtracAvg, average=average, label=label,
                           bobot=bobot,
                           number_of_training=trainingNumber, path="EigenUtil/BPPT/CROP2")

    # return bobot, eigenface, average, subtracAvg


@profile()
def testCaseDatasetLab(path="Dataset_lab/RESIZE2/CROP"):
    # sizes = [12, 25, 64, 100, 125, 320]
    # training_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    sizes = [400]
    # sizes = [12]
    training_numbers = [4, 5, 6, 7, 8, 9, 10]

    test_number = 6

    for size in sizes:
        for num in training_numbers:
            t_operasi1 = time()
            # label, average, eigenface, subtracted_average, bobot = util.getEigenUtil(path="EigenUtil/BPPT",size=size, number_of_training=num)
            label, average, eigenface, subtracted_average, bobot = util.loadTrainingFromNPZ(path="EigenUtil/BPPT/CROP2",
                                                                                            size=size,
                                                                                            number_of_training=num)
            akurasi, accuracy, recall, precision, error_rate = evaluasi.evaluasiPrecisionAndRecall(
                path=path,
                size=size,
                testnumber=test_number,
                average=average,
                eigenfaces=eigenface,
                bobot=bobot)
            t_operasi3 = time()
            t_waktu_eksekusi = t_operasi3 - t_operasi1
            nama_file = "hasil/DatasetBPPT/EigenfaceBPPT2_size" + str(size) + "_TRNum" + str(num) + "_TestNum" + str(
                test_number) + ".json"
            with open(nama_file, "w") as fp:
                fp.write("akurasi (Individu): {}\n".format(json.dumps(akurasi, indent=2)))
                # fp.write("nilai: {}\n".format(json.dumps(kl, indent=2)))
                fp.write("akurasi (ALL){}\n".format(json.dumps(accuracy)))
                fp.write("recall (ALL){}\n".format(json.dumps(recall)))
                fp.write("precision (ALL){}\n".format(json.dumps(precision)))
                fp.write("errorRate (ALL){}\n".format(json.dumps(error_rate)))
                fp.write("Durasi {}\n".format(json.dumps(t_waktu_eksekusi)))


def NONtrainingTCCustom(path="Dataset_lab/CROP"):
    sizes = [12, 25, 64, 100, 125, 320]
    # sizes = [12]
    training_numbers = [1, 2, 3, 4, 5, 6, 7, 8]
    # training_numbers = [10]
    # test_number = 6
    for size in sizes:
        print("size", size)
        for num in training_numbers:
            print("num: ", num)
            t_operasi1 = time()
            mainCustom(path=path, trainingNumber=num, size=size)
            t_operasi2 = time()
            t_waktu_computasi = t_operasi2 - t_operasi1
            profile("DURASI:", t_waktu_computasi)


def TrainingTCCustom(path="Dataset_lab/CROP"):
    # sizes = [64, 100, 125, 320, 640]
    sizes = [400]
    # sizes = [12]
    training_numbers = [4, 5, 6, 7, 8, 9, 10]
    # training_numbers = [8]
    for size in sizes:
        print("size", size)
        for num in training_numbers:
            print("num: ", num)
            t_operasi1 = time()
            mainCustom(path=path, trainingNumber=num, size=size)
            t_operasi2 = time()
            t_waktu_computasi = t_operasi2 - t_operasi1
            profile("DURASI:", t_waktu_computasi)


def testload(path="Dataset_lab/RESIZE2/CROP", size=400):
    test_number = 6
    t_operasi1 = time()
    label, average, eigenface, subtracted_average, bobot = util.loadTrainingFromNPZ(path="EigenUtil/BPPT/CROP2",
                                                                                    size=size,
                                                                                    number_of_training=10)
    akurasi, accuracy, recall, precision, error_rate = evaluasi.evaluasiPrecisionAndRecall(
        path=path,
        size=size,
        testnumber=test_number,
        average=average,
        eigenfaces=eigenface,
        bobot=bobot)
    t_operasi3 = time()
    t_waktu_eksekusi = t_operasi3 - t_operasi1
    nama_file = "hasil/DatasetBPPT/EigenfaceBPPT2_size" + str(size) + "_TRNum" + str(5) + "_TestNum" + str(
        test_number) + ".json"
    with open(nama_file, "w") as fp:
        fp.write("akurasi (Individu): {}\n".format(json.dumps(akurasi, indent=2)))
        # fp.write("nilai: {}\n".format(json.dumps(kl, indent=2)))
        fp.write("akurasi (ALL){}\n".format(json.dumps(accuracy)))
        fp.write("recall (ALL){}\n".format(json.dumps(recall)))
        fp.write("precision (ALL){}\n".format(json.dumps(precision)))
        fp.write("errorRate (ALL){}\n".format(json.dumps(error_rate)))
        fp.write("Durasi {}\n".format(json.dumps(t_waktu_eksekusi)))


def mainWebcam(path=None, size=250, channel=0):
    if path == None:
        path = "EigenUtil"

    label, average, eigenface, subtracted_average, bobot = util.loadTrainingFromNPZ(path=path, size=size)
    WebcamHandler.webcam(size=size, average=average, eigenfaces=eigenface, labels=label, bobot=bobot, channel=channel)
