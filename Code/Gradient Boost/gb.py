import os
import time
import numpy as np
from sklearn.ensemble         import GradientBoostingClassifier
from sklearn.cross_validation import KFold

def delFeatMin(arr, num):
    N = len(arr)
    sortArr = np.sort(arr)
    minElem = sortArr[0:num]
    result  = []
    for idx in range(N):
        if arr[idx] not in minElem:
            result.append(idx)
    return np.array(result)

def gbPredict(LOSS, N_EST, L_RATE, M_DEPT, SUB_S, W_START, N_FOLD, EX_F, TRAIN_DATA_X, TRAIN_DATA_Y, TEST__DATA_X, isProb):
    # feature extraction
    ### clf  = GradientBoostingClassifier(loss=LOSS, n_estimators=N_EST, learning_rate=L_RATE, max_depth=M_DEPT, subsample=SUB_S, warm_start=W_START).fit(TRAIN_DATA_X, TRAIN_DATA_Y)
    ### extA = delFeatMin(clf.feature_importances_, EX_F)
    ### TRAIN_DATA_X = TRAIN_DATA_X[:, extA]
    # k-fold validation
    kf   = KFold(TRAIN_DATA_Y.shape[0], n_folds=N_FOLD)
    tesV = 0.0
    for train_index, test_index in kf:
        X_train, X_test = TRAIN_DATA_X[train_index], TRAIN_DATA_X[test_index]
        y_train, y_test = TRAIN_DATA_Y[train_index], TRAIN_DATA_Y[test_index]
        clf  =  GradientBoostingClassifier(loss=LOSS, n_estimators=N_EST, learning_rate=L_RATE, max_depth=M_DEPT, subsample=SUB_S, warm_start=W_START).fit(X_train, y_train)
        tesK =  1 - clf.score(X_test, y_test)
        tesV += tesK
    eVal = tesV / N_FOLD
    # train all data
    clf  = GradientBoostingClassifier(loss=LOSS, n_estimators=N_EST, learning_rate=L_RATE, max_depth=M_DEPT, subsample=SUB_S, warm_start=W_START).fit(TRAIN_DATA_X, TRAIN_DATA_Y)
    TEST__DATA_X = TEST__DATA_X[:, extA]
    if isProb:
        data = clf.predict_proba(TEST__DATA_X)
    else:
        data = clf.predict(TEST__DATA_X)

    print "Eval =", eVal, "with n_esti =", N_EST, "l_rate =", L_RATE, "m_dep =", M_DEPT, "sub_s =", SUB_S, "ex_num =", EX_F, "and loss is", LOSS

    return (data, eVal)

def outputFile(INPUT_FILE, OUTPUT_FILE, data, isProb):
    outFile = open(OUTPUT_FILE, 'w')

    with open(INPUT_FILE) as inFile:
        next(inFile)
        for num, inData in enumerate(inFile, 0):
            inData = inData.rstrip()
            inData = inData.split(',')
            if isProb:
                outFile.write(inData[0] + ',' + str(data[num][1]) + '\n')
            else:
                outFile.write(inData[0] + ',' + str(data[num])    + '\n')

    inFile.close()
    outFile.close()

def main():
    # 
    N_EST_P  = 100
    N_EST_B  = 150
    L_RATE   = 0.1
    M_DEPT_P = 5
    M_DEPT_B = 6
    SUB_S    = 0.5
    N_FOLD   = 2
    EX_F_NUM = 3
    W_START  = True

    t0 = time.time() # start time

    # files path
    INPUT_FILE   = "../../ML_final_project/enrollment_test.csv"
    TRAIN_FILE_X = "../../New_Features/train_x_processed.csv"
    TRAIN_FILE_Y = "../../New_Features/train_y_processed.csv"
    TEST__FILE_X = "../../New_Features/test__x_processed.csv"
    # load files
    TRAIN_DATA_X = np.loadtxt(TRAIN_FILE_X, delimiter=',')
    TRAIN_DATA_Y = np.loadtxt(TRAIN_FILE_Y)
    TEST__DATA_X = np.loadtxt(TEST__FILE_X, delimiter=',')
    # use gradient boost classifier to predict
    (probData, eValP) = gbPredict('exponential', N_EST_P, L_RATE, M_DEPT_P, SUB_S, W_START, N_FOLD, EX_F_NUM, TRAIN_DATA_X, TRAIN_DATA_Y, TEST__DATA_X, True)
    (binaData, eValB) = gbPredict('exponential', N_EST_B, L_RATE, M_DEPT_B, SUB_S, W_START, N_FOLD, EX_F_NUM, TRAIN_DATA_X, TRAIN_DATA_Y, TEST__DATA_X, False)
    # output result
    OUTPUT_FILE1 = "../../output/gb_track1_Eval_" + str(eValP) + "_L_RATE_" + str(L_RATE) + "_N_EST_" + str(N_EST_P) + "_M_DEPT_" + str(M_DEPT_P) + "_SUB_S_" + str(SUB_S) + ".csv"
    OUTPUT_FILE2 = "../../output/gb_track2_Eval_" + str(eValB) + "_L_RATE_" + str(L_RATE) + "_N_EST_" + str(N_EST_B) + "_M_DEPT_" + str(M_DEPT_B) + "_SUB_S_" + str(SUB_S) + ".csv"
    outputFile(INPUT_FILE, OUTPUT_FILE1, probData, True)
    outputFile(INPUT_FILE, OUTPUT_FILE2, binaData, False)

    t1 = time.time() # end time
    print "...This task costs " + str(t1 - t0) + " second."

if __name__ == '__main__':
    main()