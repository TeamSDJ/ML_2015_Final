from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os


OUTPUT_FILE = "feature_importance.txt"
outFile = open(OUTPUT_FILE, 'w')


data=np.genfromtxt('train_v1.csv',dtype=str,delimiter=',')
title=data[0,1:]
title_index=np.array(list(range(len(title))),dtype=int)

serv_problem=np.argwhere(title=='server_problem')[0,0]
brow_problem=np.argwhere(title=='browser_problem')[0,0]
total_problem=np.argwhere(title=='total_problem')[0,0]

train=data[1:,1:].astype(int)#the first row is string,the first column is ID
y=np.genfromtxt('../../ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,1]

#total problem
bdt = RandomForestClassifier(n_estimators=400,max_depth=10,oob_score=True,n_jobs=-1)
bdt.fit(train,y)
imp=bdt.feature_importances_
index_order=imp.argsort()
t_title=title[index_order]
t_title_index=title_index[index_order]
outFile.write('outcome of adding total problem:'+'\n')
outFile.write('Eoob= '+str(1-bdt.oob_score_)+'\n')
for  i in range(len(title)):
	outFile.write(t_title[i]+' '+str(t_title_index[i])+'\n')

#total problem but delete server/browser problem
d_train=np.delete(train,[serv_problem,brow_problem],1)
d_title=np.delete(title,[serv_problem,brow_problem],0)
d_title_index=np.delete(title_index,[serv_problem,brow_problem],0)
bdt = RandomForestClassifier(n_estimators=400,max_depth=10,oob_score=True,n_jobs=-1)
bdt.fit(d_train,y)
imp=bdt.feature_importances_
index_order=imp.argsort()
d_title=d_title[index_order]
d_title_index=d_title_index[index_order]
outFile.write("outcome of adding total problem but delete ser/brows:"+'\n')
outFile.write('Eoob= '+str(1-bdt.oob_score_)+'\n')
for  i in range(len(d_title)):
	outFile.write(d_title[i]+' '+str(d_title_index[i])+'\n')

#without total problem(origin)
o_train=np.delete(train,total_problem,1)
o_title=np.delete(title,total_problem,0)
print(o_title)
o_title_index=np.delete(title_index,total_problem,0)
bdt = RandomForestClassifier(n_estimators=400,max_depth=10,oob_score=True,n_jobs=-1)
bdt.fit(o_train,y)
imp=bdt.feature_importances_
index_order=imp.argsort()
o_title=o_title[index_order]
o_title_index=o_title_index[index_order]
outFile.write("outcome of origin train:"+'\n')
outFile.write('Eoob= '+str(1-bdt.oob_score_)+'\n')
for  i in range(len(o_title)):
	outFile.write(o_title[i]+' '+str(o_title_index[i])+'\n')







outFile.close()

os.system('say "your program has finished"')
