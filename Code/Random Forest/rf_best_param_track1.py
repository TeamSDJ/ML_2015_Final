from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('../../ML_final_project/sample_train_x.csv',dtype='int',delimiter=',')
train=data[1:,1:]#the first row is string,the first column is ID
y=np.genfromtxt('../../ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,1]
ftr_importance=np.array([6,4,0,15,5,10,14,9,11,3,1,16,2,12,8,7,13])#importance of features

temp_train=train[:,ftr_importance[:13]]
bdt = RandomForestClassifier(n_estimators=300,max_depth=10,oob_score=True,n_jobs=-1)
bdt.fit(temp_train,y)

#test
data=np.genfromtxt('../../ML_final_project/sample_test_x.csv',dtype='int',delimiter=',')
test=data[1:,1:]
temp_test=test[:,ftr_importance[:13]]
y_eval=bdt.predict_proba(temp_test)

INPUT_FILE 	= "../../ML_final_project/enrollment_test.csv"
OUTPUT_FILE = "../../Output/rf_best_param_track1.csv"

outFile = open(OUTPUT_FILE, 'w')

with open(INPUT_FILE) as inFile:
    next(inFile)
    for num, inData in enumerate(inFile, 0):
        inData  = inData.rstrip()
        inData  = inData.split(',')
        outFile.write(inData[0] + ',' + str(y_eval[num][1]) + '\n')

inFile.close()
outFile.close()