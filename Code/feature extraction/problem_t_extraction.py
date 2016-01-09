import numpy as np
from dateutil.parser import parse
from datetime import timedelta
import matplotlib.pyplot as plt
import os
def create_t_evt_lib():
	#time and event library:lib={[event,time]}
	l_data=np.genfromtxt('ML_final_project/object.csv',dtype=str,delimiter=',')
	data=l_data[0,:]
	for s in l_data:
		if s[2]=='problem':
			data=np.vstack((data,s))
	module_id=data[1:,1]
	time=data[1:,4]
	library=np.column_stack((module_id,time))
	return library






library=np.genfromtxt('ML_final_project/problem_library.csv',dtype=str,delimiter=',')
lib_object=[]
lib_time=[]
for i in range(len(library)):
	lib_object.append(library[i,0])
	if library[i,1]=='null':
		lib_time.append(None)
	else:
		lib_time.append(parse(library[i,1]))
		

enroll_id=np.genfromtxt('ML_final_project/truth_train.csv',dtype='int',delimiter=',')[:,0]
delay_time=np.zeros(len(enroll_id),dtype=int)
ct_problem=np.zeros(len(enroll_id),dtype=int)#count the number of problems made by the user 

logtrain=np.genfromtxt('ML_final_project/log_train_problem.csv',dtype=str,delimiter=',')
event_id=[int(i) for i in logtrain[:,0]]
time=[parse(t) for t in logtrain[:,1]]
obj=[o for o in logtrain[:,2]]

logtrain=[]#recycle the memory

#extract problem time delay
prev=[]# don't need to calculate the delay time of the same problem repeatedly

for i in range(len(event_id)):
	if [event_id[i],obj[i]]!=prev:
		prev=[event_id[i],obj[i]]
		#library={[module_id,time]}
		try:#there may be some loss data,and index will raise exception
			index=lib_object.index(obj[i])
			start_time=lib_time[index]
			usr_time=time[i]
			if start_time!=None:
				en_id=np.argwhere(enroll_id==event_id[i])[0,0]
				delay_time[en_id]+=((usr_time-start_time).total_seconds())/86400#unit:day
				ct_problem[en_id]+=1
		except:
			print('data loss occur')

for i in range(len(enroll_id)):
	if ct_problem[i]!=0:
		delay_time[i]/=ct_problem[i]

OUTPUT_FILE = "delay.csv"
outFile = open(OUTPUT_FILE, 'w')
for i in range(len(delay_time)):
	outFile.write(str(delay_time[i])+','+str(ct_problem[i])+'\n')
os.system('say "your program has finished"')



















