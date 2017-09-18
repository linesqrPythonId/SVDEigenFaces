from EigenFaces import util, subspace, distance
import cv2
import numpy as np


def evaluasiPrecisionAndRecall(path="Training", size=19, testnumber=5, average=None, eigenfaces=None, bobot=None):
    # testingPict, labels = util.readImageFromPathTS(root_path=path, size=size, testNumber=testnumber)
    testingPict, labels = util.readImageFromPathTesting(root_path=path, size=size, testingNumber=testnumber)

    jumlah_citra = util.jumlahCitra(root_path=path)

    hits = 0
    noise = 0
    misses = 0

    akurasi = {}

    for key, value in testingPict.items():
        akurasi[key] = 0
        print("KEY: ", key)
        print("len(testingPict[key])", len(testingPict[key]))
        for index in range(len(testingPict[key])):
            unkOmega = subspace.getUnknownOmega(testingPict[key][index], average, eigenfaces=eigenfaces)
            nomor = distance.getClassnumber(class_label=labels, unkomega=unkOmega, testing_vec=testingPict, bobot=bobot)
            result = util.getClassName(nomor, labels)
            # print("key :", key, "\n   >result", result)
            print(" >Result: ", result)
            if result == key:
                akurasi[key] += 1
                hits += 1
            else:
                misses += 1
                noise += 1

        akurasi[key] = str(akurasi[key] / len(testingPict[key]) * 100) + str("%")
    precision = hits / (hits + noise)
    precision = precision * 100
    recall = hits / (hits + misses)
    recall = recall * 100
    accuracy = len(bobot) - (misses + noise)
    accuracy = accuracy / len(bobot)
    accuracy = (jumlah_citra) - (misses + noise)
    accuracy = accuracy / (jumlah_citra)
    error_rate = 1 - accuracy

    akurasi = [(k, akurasi[k]) for k in sorted(akurasi)]
    return akurasi, accuracy, recall, precision, error_rate


def evaluasiPrecisionAndRecallCustom(path="Training", testnumber=5, average=None, eigenfaces=None, bobot=None,
                                     jumlahcitra=10):
    testingPict, labels = util.readImageFromPathTSCustom(root_path=path, testNumber=testnumber)
    jumlah_citra = util.jumlahCitra(root_path=path)

    hits = 0
    noise = 0
    misses = 0

    akurasi = {}

    for key, value in testingPict.items():
        akurasi[key] = 0
        print("key :", key)
        for index in range(len(testingPict[key])):
            print("index")
            unkOmega = subspace.getUnknownOmega(testingPict[key][index], average, eigenfaces=eigenfaces)
            nomor = distance.getClassnumber(class_label=labels, unkomega=unkOmega, testing_vec=testingPict, bobot=bobot)
            result = util.getClassName(nomor, labels)
            print("  >result", result)
            if result == key:
                akurasi[key] += 1
                hits += 1
            else:
                misses += 1
                noise += 1

        akurasi[key] = str(akurasi[key] / len(testingPict[key]) * 100) + str("%")
    precision = hits / (hits + noise)
    precision = precision * 100
    recall = hits / (hits + misses)
    recall = recall * 100
    # accuracy = (len(bobot) * jumlahcitra) - (misses + noise)
    accuracy = (jumlah_citra) - (misses + noise)
    # accuracy = (15*11) - (misses + noise)
    # accuracy = accuracy / (len(bobot) * jumlahcitra)
    accuracy = accuracy / (jumlah_citra)
    error_rate = 1 - accuracy

    akurasi = [(k, akurasi[k]) for k in sorted(akurasi)]
    return akurasi, accuracy, recall, precision, error_rate


def webcam(size, average, eigenfaces, labels, trainingvec, bobot, channel=0):
    cap = cv2.VideoCapture(channel)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    dimension = (size, size)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not cap.isOpened:
        print("Webcam tidak dapat dijalankan")
        exit()

    while True:
        retval, frame = cap.read()

        if retval == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in face:
                cv2.rectangle(frame, (x, y), (x + w + 50, y + h + 75), (255, 0, 0), 2)
                try:
                    crop = gray[y:y + h + 10, x:x + w + 10]
                    bool_frame = True
                    imgRes = cv2.resize(crop, dimension)
                    reshapedImage = imgRes.reshape(size * size)
                    result = __webcamHandler_(reshapedImage, average, eigenfaces, labels, trainingvec, bobot)
                    cv2.putText(frame, "result: " + str(result), (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                                cv2.LINE_AA)
                except Exception as e:
                    print("Err:" + str(e))
                    print("Kepala kurang ke atas")
                    cv2.putText(frame, "Kepala kurang ke atas", (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                                cv2.LINE_AA)
                    bool_frame = False
                    # if bool_frame == True:
                    #     imgRes = cv2.resize(crop, dimension)
                    #     reshapedImage = imgRes.reshape(size * size)
                    #     result = __webcamHandler_(reshapedImage, average, eigenfaces, labels, trainingvec, bobot)
                    #     cv2.putText(frame, "result: " + str(result), (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                    #                 cv2.LINE_AA)
                    #     print("result: " + result)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def __webcamHandler_(image, average, eigenfaces, labels, trainingvec, bobot):
    unkOmega = subspace.getUnknownOmega(test_image=image, average=average, eigenfaces=eigenfaces)
    nomor = distance.getClassnumber(class_label=labels, unkomega=unkOmega, testing_vec=labels, bobot=bobot)
    result = util.getClassName(nomor, labels)

    return result
