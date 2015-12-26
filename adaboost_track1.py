from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data=np.genfromtxt('sample_train_x.csv',dtype='int',delimiter=',')
train=data[1:,:]#the first row is string
y=np.genfromtxt('truth_train.csv',dtype='int',delimiter=',')[:,1]

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(train,y)

#output the evaluattion of testing
data=np.genfromtxt('sample_test_x.csv',dtype='int',delimiter=',')
test=data[1:,:]#the first row is string
y_eval=bdt.predict_proba(test)

INPUT_FILE 	= "ML_final_project/enrollment_test.csv"
OUTPUT_FILE = "Output/adaboost_tree_track1.csv"

outFile = open(OUTPUT_FILE, 'w')

with open(INPUT_FILE) as inFile:
    next(inFile)
    for num, inData in enumerate(inFile, 0):
        inData  = inData.rstrip()
        inData  = inData.split(',')
        outFile.write(inData[0] + ',' + str(y_eval[num][1]) + '\n')

inFile.close()
outFile.close()