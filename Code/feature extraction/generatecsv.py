import numpy as np
import os

#generate problem
logtrain=np.genfromtxt('ML_final_project/log_train.csv',dtype=str,delimiter=',')[1:,:]
OUTPUT_FILE = "ML_final_project/log_train_problem.csv"
outFile = open(OUTPUT_FILE, 'w')

for log in logtrain:
	if log[3]=='problem':
		outFile.write(log[0]+','+log[1]+',' +log[4]+'\n')


outFile.close()
os.system('say "your program has finished"')
