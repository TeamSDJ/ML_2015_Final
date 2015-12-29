import os
import numpy as np

INPUT_FILE 	= "../../ML_final_project/enrollment_test.csv"
OUTPUT_FILE = "../../Output/random_guess_track2.csv"

outFile = open(OUTPUT_FILE, 'w')

with open(INPUT_FILE) as inFile:
    next(inFile)
    for inData in inFile:
        inData  = inData.rstrip()
        inData  = inData.split(',')
        randNum = np.random.random_sample()
        if randNum >= 0.5:
            randNum = 1
        else:
            randNum = 0
        outFile.write(inData[0] + ',' + str(randNum) + '\n')

inFile.close()
outFile.close()