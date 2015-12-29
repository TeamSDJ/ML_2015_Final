from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data=np.genfromtxt('../../ML_final_project/sample_train_x.csv',dtype='int',delimiter=',')
train=data[1:,1:]#the first row is string,the first column is ID
y=np.genfromtxt('../../ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,1]

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                         algorithm="SAMME",
                         n_estimators=200)
bdt.fit(train,y)
print('score is: ',bdt.score(train,y))#score=accuracy,not error rate
output the evaluattion of testing
data=np.genfromtxt('../../ML_final_project/sample_test_x.csv',dtype='int',delimiter=',')
test=data[1:,1:]#the first row is string,the first column is ID
y_eval=bdt.predict_proba(test)

INPUT_FILE 	= "../../ML_final_project/enrollment_test.csv"
OUTPUT_FILE = "../../Output/adaboost_tree_track1.csv"

outFile = open(OUTPUT_FILE, 'w')

with open(INPUT_FILE) as inFile:
    next(inFile)
    for num, inData in enumerate(inFile, 0):
        inData  = inData.rstrip()
        inData  = inData.split(',')
        outFile.write(inData[0] + ',' + str(y_eval[num][1]) + '\n')

inFile.close()
outFile.close()