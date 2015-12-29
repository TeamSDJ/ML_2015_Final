from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

data=np.genfromtxt('../../ML_final_project/sample_train_x.csv',dtype='int',delimiter=',')
train=data[1:,1:]#the first row is string,the first column is ID
y=np.genfromtxt('../../ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,1]

#the parameters needing to be determined
max_depth=[10,20,30,40,None]
ftr_importance=np.array([6,4,0,15,5,10,14,9,11,3,1,16,2,12,8,7,13])#importance of features
n_feature=range(10,17,1)#number of feature used
E_oob=[]
param=[]
#determine the parameters using the oob
for d in max_depth:
	for f in n_feature:
		temp_train=train[:,ftr_importance[:f]]
		bdt = RandomForestClassifier(n_estimators=300,max_depth=d,oob_score=True,n_jobs=-1)
		bdt.fit(temp_train,y)
		E_oob.append(1-bdt.oob_score_)
		param.append([d,f])

print('minimum oob is:',min(E_oob))
param_choose=param[E_oob.index(min(E_oob))]
print('the param is: ',param_choose)









#output the evaluattion of testing
#data=np.genfromtxt('sample_test_x.csv',dtype='int',delimiter=',')
#test=data[1:,:]#the first row is string
#y_eval=bdt.predict(test)

#INPUT_FILE 	= "ML_final_project/enrollment_test.csv"
#OUTPUT_FILE = "Output/adaboost_tree_track2.csv"

#outFile = open(OUTPUT_FILE, 'w')

#with open(INPUT_FILE) as inFile:
#    next(inFile)
#    for num, inData in enumerate(inFile, 0):
#        inData  = inData.rstrip()
#        inData  = inData.split(',')
#        outFile.write(inData[0] + ',' + str(y_eval[num]) + '\n')

#inFile.close()
#outFile.close()
