import numpy as np
from dateutil.parser import parse
from datetime import timedelta
import matplotlib.pyplot as plt
import os

#generate library
library=np.genfromtxt('video_library.csv',dtype=str,delimiter=',')
lib_object=[]
lib_time=[]
for i in range(len(library)):
	lib_object.append(library[i,0])
	if library[i,1]=='null':
		lib_time.append(None)
	else:
		lib_time.append(parse(library[i,1]))
os.system('say "library finished"')

enroll_id=np.genfromtxt('../../ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,0]
delay_time=np.zeros(len(enroll_id),dtype=int)
recorded_v=[[]]*len(enroll_id)#recode the viewed video
ct_problem=np.zeros(len(enroll_id),dtype=int)
#generate log video data
logtrain=np.genfromtxt('log_train_video.csv',dtype=str,delimiter=',')
event_id=[int(i) for i in logtrain[:,0]]
time=[parse(t) for t in logtrain[:,1]]
obj=[o for o in logtrain[:,2]]

logtrain=[]#recycle the memory
os.system('say "log finished"')
#calculate delay
for i in range(len(event_id)):
	en_id=np.argwhere(enroll_id==event_id[i])[0,0]
	if obj[i] not in recorded_v[en_id]:#not being viewed
		recorded_v[en_id].append(obj[i])
		#library={[module_id,time]}
		try:#there may be some loss data,and index will raise exception
			index=lib_object.index(obj[i])
			start_time=lib_time[index]
			usr_time=time[i]
			if start_time!=None:
				delay_time[en_id]+=((usr_time-start_time).total_seconds())/86400#unit:day
				ct_problem[en_id]+=1
		except:
			print('data loss occur')
for i in range(len(enroll_id)):
	if ct_problem[i]!=0:
		delay_time[i]/=ct_problem[i]

OUTPUT_FILE = "delay_video.csv"
outFile = open(OUTPUT_FILE, 'w')
for i in range(len(delay_time)):
	outFile.write(str(delay_time[i])+','+str(ct_problem[i])+'\n')
outFile.close()
os.system('say "your program has finished"')





