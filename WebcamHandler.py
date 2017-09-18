import cv2
from EigenFaces import subspace, util, distance


def webcam(size, average, eigenfaces, labels, bobot, channel=0):
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
                    result = __webcamHandler_(reshapedImage, average, eigenfaces, labels, bobot)
                    cv2.putText(frame, "result: " + str(result), (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                                cv2.LINE_AA)
                except Exception as e:
                    print("Err:" + str(e))
                    print("Kepala kurang ke atas")
                    cv2.putText(frame, "Kepala kurang ke atas", (x + w, y + h), font, 0.5, (0, 247, 255), 2,
                                cv2.LINE_AA)
                    bool_frame = False
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def __webcamHandler_(image, average, eigenfaces, labels, bobot):
    unkOmega = subspace.getUnknownOmega(test_image=image, average=average, eigenfaces=eigenfaces)
    nomor = distance.getClassnumber(class_label=labels, unkomega=unkOmega, testing_vec=labels, bobot=bobot)
    result = util.getClassName(nomor, labels)

    return result
