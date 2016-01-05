import os
import time
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

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
    C_CONST = 100
    GAMMA   = 1.12
    KERNEL  = 'rbf'

    t0 = time.time()

    INPUT_FILE   = "../../ML_final_project/enrollment_test.csv"
    OUTPUT_FILE  = "../../Output/svm_track2_K_" + KERNEL + "_C_" + str(C_CONST) + "_Gamma_" + str(GAMMA) + ".csv"

    TRAIN_FILE_X = "../../ML_final_project/sample_train_x.csv"
    TRAIN_FILE_Y = "../../ML_final_project/truth_train.csv"

    TRAIN_DATA_X = np.loadtxt(TRAIN_FILE_X, delimiter=',', skiprows=1, usecols=range(1, 18))
    TRAIN_DATA_Y = np.loadtxt(TRAIN_FILE_Y, delimiter=',')

    TEST_FILE_X  = "../../ML_final_project/sample_test_x.csv"
    TEST_DATA_X  = np.loadtxt(TEST_FILE_X, delimiter=',', skiprows=1, usecols=range(1, 18))

    scaler = StandardScaler()
    trainX = scaler.fit_transform(TRAIN_DATA_X)
    trainY = TRAIN_DATA_Y[:, (TRAIN_DATA_Y.shape[1] - 1)]
    testX  = scaler.transform(TEST_DATA_X)

    t1 = time.time()
    print "...scaling costs " + str(t1 - t0) + " second, start training model..."

    clf  = svm.SVC(C=C_CONST, kernel=KERNEL, gamma=GAMMA)
    clf.fit(trainX, trainY)

    t2 = time.time()
    print "...training costs " + str(t2 - t1) + " second, start predicting test data..."

    data = clf.predict(testX)
    outputFile(INPUT_FILE, OUTPUT_FILE, data)

    t3 = time.time()
    print "...predicting costs " + str(t3 - t2) + " second."
    print ""
    print "This task costs " + str(t3 - t0) + " second."

if __name__ == '__main__':
    main()