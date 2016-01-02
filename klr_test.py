
import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt
#XXX read in the files

#XXX First reload the data we generated in 1_notmist.ipynb.

lemda = 0.5
gama = 1000000000
train_size = 15000
val_size = 15000
data = np.matrix(np.genfromtxt('sample_train_x.txt', delimiter=',')[1:,1:])
truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])
test_data = np.matrix(np.genfromtxt('sample_test_x.txt', delimiter=',')[1:,1:])
test_data = test_data/test_data.sum(axis=0)

data = data/data.sum(axis=0)
truth = truth*2-1
train_x = data[:train_size,:]
train_y = truth[:train_size,:]

np.shape(train_x)
np.shape(train_y)

val_x = test_data

dim = train_x.shape[1]
N = train_x.shape[0]


M = val_x.shape[0]

print "train_size = ",N," val size = ",M
#target = 1+2*np.random.randint(-1,high=1,size = (N,1))


with tf.device('/cpu:0'):
    #XXX setup the tf const for the training data

    x = tf.constant(train_x,dtype=tf.float32) #(N,dim)

    #y_v = tf.constant(val_y,dtype=tf.float32)
    x_v = tf.constant(val_x,dtype=tf.float32)

    #XXX formating the kernel of validation data: (M,N)
    xm_2 = tf.mul(x_v,x_v) #(M,d)
    xn_2 = tf.mul(x,x) #(N,d)

    xxTmn = tf.matmul(x_v,tf.transpose(x,perm=(1,0))) #(M,N)

    onesMd = tf.constant(np.ones((M,dim)),dtype=tf.float32) #(M,d)
    onesNd = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(M,d)
    xxs_xm = tf.matmul(onesNd,tf.transpose(xm_2,perm=(1,0)))
    xxs_xn = tf.matmul(onesMd,tf.transpose(xn_2,perm=(1,0)))

    val_tmp1 = tf.add(xxs_xn,tf.transpose(xxs_xm,perm=(1,0)))
    val_tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxTmn)

    val_kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(val_tmp1,val_tmp2)))


with tf.device('/cpu:0'):
    #XXX setup the variables
    betas = tf.Variable(tf.zeros([N, 1],dtype=tf.float32))
    #XXX kernel placeholder
    val_kernel_holder = tf.placeholder(tf.float32,shape=(M,N))

with tf.device('/cpu:0'):

    val_prediction = tf.sign(tf.matmul(val_kernel_holder,betas))
saver = tf.train.Saver()

with tf.Session() as session:
    val_kernel_variable = val_kernel.eval()
print "validation kernel complete!"
#np.shape(test_kernel_variable)
num_steps = 1
with tf.Session() as session:

    saver.restore(session,"klr_model.ckpt")
    for step in range(num_steps):
        if step%1==0:
	    tp= session.run(val_prediction, feed_dict={val_kernel_holder:val_kernel_variable})
        print tp

tp =(tp+1)/2
result_file = open('result.txt','w')
for i in range(tp.shape[0]):
    result_file.write(str(int(tp[i,0]))+'\n')
result_file.close()
