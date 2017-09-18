import numpy as np
from numba import jit


def projectWeight(subtracted_average=None, eigenfaces=None):
    if subtracted_average.all() == None:
        if eigenfaces.all() == None:
            print("Item yang diberikan tidak tersedia")
            return None

    weightfaces = []
    for index in range(0, len(subtracted_average[0])):
        weight = []
        for value in eigenfaces:
            weight.append(np.dot(value.T, np.transpose(subtracted_average)[index]))
        weightfaces.append(weight)
    return weightfaces


# @jit
def computeEigenSmall(covariance_matrix=None, subtracted_average=None):
    U, s, V = np.linalg.svd(covariance_matrix, full_matrices=True, compute_uv=True)
    # U, s, V = np.linalg.svd(subtracted_average, full_matrices=True, compute_uv=True)
    V = np.transpose(V)
    print("U", np.shape(U))
    print("V", np.shape(V))
    print("s", np.shape(s))
    eigenValuesTemp = s ** -0.5
    print("eigenValuesTemp", np.shape(eigenValuesTemp))
    eigenVectorU = []
    # for i in range(len(V)):
    count = 0
    for values in V:
        # print("values", np.shape(values))
        try:
            matrix_multiplied = np.dot(subtracted_average, values)
        except Exception as e:
            matrix_multiplied = np.dot(np.transpose(subtracted_average), values)

        eigenVectorU.append(matrix_multiplied / eigenValuesTemp[count])
        count += 1

    return eigenValuesTemp, eigenVectorU


def getUnknownOmega(test_image, average, eigenfaces):
    unkAvg = test_image - average
    unkOmega = []
    for value in eigenfaces:
        unkOmega.append(np.dot(np.transpose(value), unkAvg))
    return unkOmega
