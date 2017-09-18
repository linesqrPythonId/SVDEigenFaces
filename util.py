import numpy as np
import os
import glob
import cv2
from numba import jit
from memory_profiler import profile


def __getCropppedFace(gray_scale_image, yPlus=0, yMin=0, xMin=0, xPlus=0, face_cascade=None):
    if gray_scale_image.any() == None:
        return None
    if face_cascade == None:
        return None

    d = face_cascade.detectMultiScale(gray_scale_image, 1.3, 5)
    try:
        [[x, y, w, h]] = d
        crop = gray_scale_image[y - yMin:y + h + yPlus, x - xMin:x + w + xPlus]
        return crop, np.shape(crop)

    except Exception as e:
        print("Err: ", str(e))

    return "Salah", None


def readImageFromPath(root_path, size_reduction=2, file_extension=".jpg", size=10):
    # dimension = (size, size)
    labels = []
    image_vectors = []
    for i in os.listdir(root_path):
        labels.append(i)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    dimension = (size, size)

    for tr_path in labels:
        print("path: ", tr_path)
        count = 0
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            if count >= 1:
                break
            count += 1
            img = cv2.imread(pathImage, 0)
            crop, shape = __getCropppedFace(img, yPlus=10, xPlus=10, face_cascade=face_cascade)
            # imgRes = cv2.resize(crop, (int(shape[1] / size_reduction), int(shape[0] / size_reduction)))
            imgRes = cv2.resize(crop, dimension)
            # shape_ = np.shape(imgRes)
            imgVec = imgRes.reshape(size * size)
            image_vectors.append(imgVec)

    return np.asarray(image_vectors), labels


def readImageFromPathTrainingWithCrop(root_path, file_extension=".pgm", trainingNumber=10):
    labels = []
    image_vectors = []
    for i in os.listdir(root_path):
        labels.append(i)

    for tr_path in labels:
        print("path: ", tr_path)
        count = 0
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            if count >= trainingNumber:
                break
            count += 1
            img = cv2.imread(pathImage, -1)
            shape = np.shape(img)
            img2 = cv2.resize(img, (int(shape[0] / 1), int(shape[1] / 1)))
            shape = np.shape(img2)
            imgVec = img2.reshape(shape[0] * shape[1])
            image_vectors.append(imgVec)

    return np.asarray(image_vectors), labels


def readImageFromPathTestWithCrop(root_path, file_extension=".jpg", size=10, testNumber=10):
    labels = []
    image_vectors = {}
    for i in os.listdir(root_path):
        labels.append(i)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    dimension = (size, size)
    for tr_path in labels:
        print("path: ", tr_path)
        count = 0
        vectortest = []
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            count += 1
            if count >= (10 - testNumber):
                break
            img = cv2.imread(pathImage, 0)
            crop, shape = __getCropppedFace(img, yPlus=10, xPlus=10, face_cascade=face_cascade)
            imgRes = cv2.resize(crop, dimension)
            imgVec = imgRes.reshape(size * size)
            vectortest.append(imgVec)
        image_vectors[tr_path] = vectortest

    return image_vectors, labels


def readImageFromPathTSCustom(root_path, file_extension=".pgm", testNumber=2):
    labels = []
    image_vectors = {}
    for i in os.listdir(root_path):
        labels.append(i)

    for tr_path in labels:
        print("path: ", tr_path)
        count = 0
        vectortest = []
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            count += 1
            if count >= (11 - (testNumber - 1)):
                break
            img = cv2.imread(pathImage, -1)
            shape = np.shape(img)
            img2 = cv2.resize(img, (int(shape[0] / 1), int(shape[1] / 1)))
            shape = np.shape(img2)
            imgVec = img2.reshape(shape[0] * shape[1])
            vectortest.append(imgVec)
        image_vectors[tr_path] = vectortest

    return image_vectors, labels


def readImageFromPathTraining(root_path, file_extension=".jpg", size=10, trainingNumber=10):
    labels = []
    image_vectors = []
    for i in os.listdir(root_path):
        labels.append(i)

    dimension = (size, size)
    for tr_path in labels:
        print("path: ", tr_path)
        count = 0
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            if count >= trainingNumber:
                break
            count += 1
            img = cv2.imread(pathImage, 0)
            imgRes = cv2.resize(img, dimension)
            imgVec = imgRes.reshape(size * size)
            image_vectors.append(imgVec)

    return np.asarray(image_vectors), labels


def readImageFromPathTesting(root_path, file_extension=".jpg", size=10, testingNumber=10):
    labels = []
    image_vectors = {}
    for i in os.listdir(root_path):
        labels.append(i)

    dimension = (size, size)
    for tr_path in labels:
        print("path Test: ", tr_path)
        count = 0
        vectortest = []
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            count += 1
            if count >= (11 - (testingNumber - 1)):
                img = cv2.imread(pathImage, 0)
                imgRes = cv2.resize(img, dimension)
                imgVec = imgRes.reshape(size * size)
                vectortest.append(imgVec)
        image_vectors[tr_path] = vectortest

    print("image_vectors", np.shape(image_vectors))
    return image_vectors, labels


def computeAverage(image_vectors):
    average = 0
    for value in image_vectors:
        average += value
    average = average / len(image_vectors[1])
    return average


def computeSubtractedAverage(image_vectors, average):
    subtracted_average = []
    for value in image_vectors:
        subtracted_average.append(value - average)
    return np.transpose(subtracted_average)


def computeCovarianceMatrix(subtracted_average):
    if subtracted_average.all() == None:
        return None

    x, y = np.shape(subtracted_average)
    if y > x:
        matCov = np.dot(subtracted_average, np.transpose(subtracted_average))
        return matCov
    elif x > y:
        matCov = np.dot(np.transpose(subtracted_average), subtracted_average)
        return matCov


@profile
def centralization(eigenVector):
    sum_of_squares = 0
    for i in range(len(eigenVector)):
        for value in eigenVector[i]:
            sum_of_squares += value ** 2
        vectorAvg = np.sqrt(sum_of_squares)
        eigenVector[i] = eigenVector[i] / vectorAvg
    return eigenVector


@jit
def normalization(eigenVector):
    print("normalisasi")
    eigenVector = np.asarray(eigenVector)
    maxPixel = eigenVector.max()
    minPixel = eigenVector.min()
    for index in range(len(eigenVector)):
        new_pixel = []
        for indexPixel in range(len(eigenVector[index])):
            value = 255 * (eigenVector[index, indexPixel] - minPixel) / (maxPixel - minPixel)

            value = np.uint8(value.real)
            new_pixel.append(value)
        eigenVector[index] = new_pixel
    return eigenVector


def setEigenface(kEigenfaces, eigenvector):
    return eigenvector[:kEigenfaces]


def getWeight(eigenFaces, subtracted_average):
    weight = []
    for index in range(len(subtracted_average[0])):
        weight_temp = []
        for value in eigenFaces:
            rubah = np.transpose(np.reshape(value, (-1, 1)))
            try:
                weight_temp.append(
                    np.dot(rubah, np.transpose(subtracted_average)[index]))
            except Exception as e:
                weight_temp.append(np.dot(rubah, np.transpose(np.transpose(subtracted_average)[index])))
        weight.append(weight_temp)
    return weight


def saveTrainingToNPZ(path="Eigenutil", eigenFaces=None, subtracted_average=None, average=None, label=None, bobot=None,
                      number_of_training=1):
    size = int(np.sqrt(len(subtracted_average)))
    nama_file = path + "/eigenTraining" + "_" + str(size) + "_" + "_TR_" + str(number_of_training)
    try:
        np.savez_compressed(file=nama_file, eigenFaces=eigenFaces, subtracted_average=subtracted_average,
                            average=average, label=label, bobot=bobot)
        print("Tersimpan")
    except Exception as e:
        print("Erro: ", e)
        raise


def loadTrainingFromNPZ(path="Eigenutil", size=250, number_of_training=1):
    nama_file = path + "/eigenTraining" + "_" + str(size) + "_" + "_TR_" + str(number_of_training) + ".npz"

    try:
        dtmp = np.load(nama_file)
        eigenFaces = dtmp["eigenFaces"]
        subtracted_average = dtmp["subtracted_average"]
        average = dtmp["average"]
        label = dtmp["label"]
        bobot = dtmp["bobot"]
        return label, average, eigenFaces, subtracted_average, bobot
    except Exception as e:
        print("Error: ", str(e))
        raise


def loadTrainingFromNPZWeb(nama_file="Eigenutil"):
    try:
        dtmp = np.load(nama_file)
        eigenFaces = dtmp["eigenFaces"]
        subtracted_average = dtmp["subtracted_average"]
        average = dtmp["average"]
        label = dtmp["label"]
        bobot = dtmp["bobot"]
        size = int(np.sqrt(len(subtracted_average)))
        return label, average, eigenFaces, subtracted_average, bobot, size
    except Exception as e:
        print("Error: ", str(e))
        raise


def getClassName(nomor, labels):
    # print("nomor", nomor)
    return labels[nomor]


def saveCrop(path, image):
    print("path:", path)
    cv2.imwrite(path, image)


def jumlahCitra(root_path):
    labels = []
    for i in os.listdir(root_path):
        labels.append(i)
    count = 0
    for path in labels:
        print("path:", path)
        for imagePath in glob.glob(root_path + "/" + path + "/*.jpg"):
            print("> ", imagePath)
            count += 1
    return count


def cropAndSave(root_path, file_extension=".jpg"):
    labels = []
    for i in os.listdir(root_path):
        labels.append(i)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for tr_path in labels:
        print("path Test: ", tr_path)
        for pathImage in glob.glob(root_path + "/" + tr_path + "/*" + file_extension):
            img = cv2.imread(pathImage, 0)
            # CROP
            crop, shape = __getCropppedFace(img, yPlus=10, xPlus=10, face_cascade=face_cascade)

            if type(crop) == str:
                print("     >", pathImage, " TIDAK TERBACA")
                continue
            else:
                # pass
                saveCrop(path=pathImage + "crop.jpg", image=crop)
            print("     >", pathImage)
