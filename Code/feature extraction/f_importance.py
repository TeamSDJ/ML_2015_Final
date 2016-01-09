from sklearn.ensemble import RandomForestClassifier
import numpy as np

data=np.genfromtxt('ML_final_project/sample_train_x.csv',dtype='int',delimiter=',')
train=data[1:,1:]#the first row is string,the first column is ID
y=np.genfromtxt('ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,1]
bdt = RandomForestClassifier(n_estimators=2000,max_depth=10,oob_score=True,n_jobs=-1)
bdt.fit(train,y)
print(bdt.feature_importances_)