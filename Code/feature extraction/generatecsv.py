import numpy as np
import os

#generate problem
'''
logtrain=np.genfromtxt('../../ML_final_project/log_train.csv',dtype=str,delimiter=',')[1:,:]
OUTPUT_FILE = "log_train_problem.csv"
outFile = open(OUTPUT_FILE, 'w')

for log in logtrain:
	if log[3]=='problem':
		outFile.write(log[0]+','+log[1]+',' +log[4]+'\n')


outFile.close()
'''
#generate video
'''
logtrain=np.genfromtxt('../../ML_final_project/log_train.csv',dtype=str,delimiter=',')[1:,:]
OUTPUT_FILE = "log_train_video.csv"
outFile = open(OUTPUT_FILE, 'w')

for log in logtrain:
	if log[3]=='video':
		outFile.write(log[0]+','+log[1]+',' +log[4]+'\n')
outFile.close()
#generate videolibrary
obj=np.genfromtxt('../../ML_final_project/object.csv',dtype=str,delimiter=',')[1:,:]
OUTPUT_FILE = "video_library.csv"
outFile = open(OUTPUT_FILE, 'w')

for o in obj:
	if o[2]=='video':
		outFile.write(o[1]+','+o[4]+'\n')
outFile.close()
'''
#combine server and browser into one feature
train=np.genfromtxt('../../ML_final_project/sample_train_x.csv',dtype=str,delimiter=',')
OUTPUT_FILE = "train_v1.csv"
outfile = open(OUTPUT_FILE, 'w')
#deal with the first row(feature name)
for t in train[0,:]:
	outfile.write(t+',')
outfile.write('total_problem'+'\n')
serv_problem=np.argwhere(train[0,:]=='server_problem')[0,0]
brow_problem=np.argwhere(train[0,:]=='browser_problem')[0,0]

train=train[1:,:]
for t in train:
	for elem_t in t:
		outfile.write(elem_t+',')
	outfile.write(str(int(t[serv_problem])+int(t[brow_problem]))+'\n')
outfile.close()
















os.system('say "your program has finished"')


