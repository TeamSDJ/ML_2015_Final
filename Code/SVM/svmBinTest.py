import os
import time
import numpy as np
from sklearn import svm

def outputFile(INPUT_FILE, OUTPUT_FILE, data):
    outFile = open(OUTPUT_FILE, 'w')

    with open(INPUT_FILE) as inFile:
        next(inFile)
        for num, inData in enumerate(inFile, 0):
            inData = inData.rstrip()
            inData = inData.split(',')
            outFile.write(inData[0] + ',' + str(data[num]) + '\n')

    inFile.close()
    outFile.close()

def main():
    C_CONST = 1.1
    GAMMA   = 10

    INPUT_FILE   = "../../ML_final_project/enrollment_test.csv"
    OUTPUT_FILE  = "../../Output/svm_test_track2_C_" + str(C_CONST) + "_Gamma_" + str(GAMMA) + ".csv"
    TRAIN_FILE_X = "../../ML_final_project/sample_train_x.csv"
    TRAIN_FILE_Y = "../../ML_final_project/truth_train.csv"
    TRAIN_DATA_X = np.loadtxt(TRAIN_FILE_X, delimiter=',', skiprows=1, usecols=range(1, 18))
    TRAIN_DATA_Y = np.loadtxt(TRAIN_FILE_Y, delimiter=',')

    TEST_FILE_X  = "../../ML_final_project/sample_test_x.csv"
    TEST_DATA_X  = np.loadtxt(TEST_FILE_X, delimiter=',', skiprows=1, usecols=range(1, 18))

    t0 = time.time()

    trainX = TRAIN_DATA_X
    trainY = TRAIN_DATA_Y[:, (TRAIN_DATA_Y.shape[1] - 1)]

    testX  = TEST_DATA_X

    clf  = svm.SVC(C=C_CONST, kernel='rbf', gamma=GAMMA)
    clf.fit(trainX, trainY)
    data = clf.predict(testX)

    outputFile(INPUT_FILE, OUTPUT_FILE, data)

    t1 = time.time()
    print "...costs " + str(t1 - t0) + " second."

if __name__ == '__main__':
    main()