import os
import time
import numpy as np
from sklearn.preprocessing    import StandardScaler

def logFileTimeCount(ORI_DATA):
    DATA    = np.bincount(ORI_DATA[:, 0].astype(int))
    DATA_ID = np.nonzero(DATA)[0]
    D_COUNT = np.vstack((DATA_ID, DATA[DATA_ID])).T[:, 1]

    timeTot = []
    cursor  = 0
    for idx in range(len(D_COUNT)):
        timeArr = []
        enrollCount = D_COUNT[idx]
        for row in range(cursor, (cursor + enrollCount)):
            timeArr.append(np.datetime64(ORI_DATA[row][1]))
        cursor    += enrollCount
        startTime =  min(timeArr)
        logCount  =  0
        for time in timeArr:
            if (time - startTime).item().total_seconds() >= (86400 * 25):
                logCount += 1
        D_COUNT[idx] = logCount
        timeTot.append((max(timeArr) - min(timeArr)).item().total_seconds())
    return np.column_stack((D_COUNT, np.array(timeTot)))

def outputXFile(INPUT_FILE, OUTPUT_FILE, DATA):
    outFile = open(OUTPUT_FILE, 'w')
    outFile.write('ID,user_log_num,course_log_num,take_course_num,take_user_num,log_num,server_nagivate,server_access,server_problem,browser_access,browser_problem,browser_page_close,browser_video,server_discussion,server_wiki,chapter_count,sequential_count,video_count,avg_log_num,log_event_after_25,log_total_time')
    outFile.write('\n')
    idArr = []
    with open(INPUT_FILE) as inFile:
        next(inFile)
        for num, inData in enumerate(inFile, 0):
            inData = inData.rstrip()
            inData = inData.split(',')
            idArr.append(inData[0])
    for row in range(DATA.shape[0]):
        colStr = str(idArr[row]) + ','
        for column in range(DATA.shape[1]):
            if column == (DATA.shape[1] - 1):
                colStr += (str(DATA[row][column]) + '\n')
            else:
                colStr += (str(DATA[row][column]) + ',')
        outFile.write(colStr)

    outFile.close()

def outputYFile(OUTPUT_FILE, DATA):
    outFile = open(OUTPUT_FILE, 'w')
    for idx in range(len(DATA)):
        outFile.write(str(DATA[idx]) + '\n')
    outFile.close()

def main():
    
    t0 = time.time() # start time

    # output files path
    TRAINX_OUTPUT = "../../New_Features/train_x_processed.csv"
    TEST_X_OUTPUT = "../../New_Features/test__x_processed.csv"
    # input files path
    INPUT_FILE_1  = "../../ML_final_project/enrollment_train.csv"
    INPUT_FILE_2  = "../../ML_final_project/enrollment_test.csv"
    TRAIN_FILE_X1 = "../../ML_final_project/train_v2.csv"
    TRAIN_FILE_X2 = "../../ML_final_project/log_train.csv"
    TEST__FILE_X1 = "../../ML_final_project/test_v2.csv"
    TEST__FILE_X2 = "../../ML_final_project/log_test.csv"
    # load files
    TRAIN_DATA_X1 = np.loadtxt(TRAIN_FILE_X1, delimiter=',', skiprows=1, usecols=(range(1, 19)))
    TEST__DATA_X1 = np.loadtxt(TEST__FILE_X1, delimiter=',', skiprows=1, usecols=(range(1, 19)))
    TRAIN_DATA_X2 = logFileTimeCount(np.loadtxt(TRAIN_FILE_X2, delimiter=',', skiprows=1, dtype=object))
    TEST__DATA_X2 = logFileTimeCount(np.loadtxt(TEST__FILE_X2, delimiter=',', skiprows=1, dtype=object))
    # combine files
    TRAIN_DATA_X0 = np.column_stack((TRAIN_DATA_X1, TRAIN_DATA_X2))
    TEST__DATA_X0 = np.column_stack((TEST__DATA_X1, TEST__DATA_X2))
    # data preprocessing
    scaler = StandardScaler()
    TRAIN_DATA_X = scaler.fit_transform(TRAIN_DATA_X0)
    TEST__DATA_X = scaler.transform(TEST__DATA_X0)
    # output processed files
    outputXFile(INPUT_FILE_1, TRAINX_OUTPUT, TRAIN_DATA_X)
    outputXFile(INPUT_FILE_2, TEST_X_OUTPUT, TEST__DATA_X)

    t1 = time.time() # end time
    print "...This task costs " + str(t1 - t0) + " second."

if __name__ == '__main__':
    main()