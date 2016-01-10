# NOTE 2016/1/2 :
#  - In this program, I do the prediction of testing data by the KLR model,
# with the beta variables stored in klr_model.ckpt.
# - I try to split the testing kernel into two so that the memory of GPU are able to
# handle the tranmendous amount of testing data!


import numpy as np
import tensorflow as tf
import time
import sys

#XXX read in the files

#XXX First reload the data we generated in 1_notmist.ipynb.

# NOTE : the model const lemda and kernel const are choosen by try-and-error
lemda = 0.5
gama = 1e1

# NOTE : here we use the first 15000 training data to calculate the training-testing data!
train_size = 15000
data = np.matrix(np.genfromtxt('train_x_processed.txt', delimiter=',')[1:,1:])
truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])

data = data/data.sum(axis=0) #NOTE: do the normalization
train_x = data[:train_size,:]
train_y = truth[:train_size,:]


test_data = np.matrix(np.genfromtxt('test_x_processed.txt', delimiter=',')[1:,1:])
test_data = test_data/test_data.sum(axis=0) #NOTE : do the normalization


# NOTE: splitting the testing data
test_x1 = test_data[:15000,:]
test_x2 = test_data[15000:,:]
dim = train_x.shape[1]
N = train_x.shape[0]

M1 = test_x1.shape[0]
M2 = test_x2.shape[0]
print "train_size = ",N," test size = ",M1+M2

with tf.device('/cpu:0'):
    #XXX setup the variables
    betas = tf.Variable(tf.zeros([N, 1],dtype=tf.float32))
    #XXX kernel placeholder


# the kernel for the 1st part test data
with tf.device('/cpu:0'):
    #XXX setup the tf const for the training data

    x = tf.constant(train_x,dtype=tf.float32) #(N,dim)
    x_t1 = tf.constant(test_x1,dtype=tf.float32)

    #XXX formating the kernel of validation data: (M,N)
    xm1_2 = tf.mul(x_t1,x_t1) #(M,d)
    xn_2 = tf.mul(x,x) #(N,d)

    xxTm1n = tf.matmul(x_t1,tf.transpose(x,perm=(1,0))) #(M,N)

    onesM1d = tf.constant(np.ones((M1,dim)),dtype=tf.float32) #(M,d)
    onesNd = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(M,d)
    xxs_xm1 = tf.matmul(onesNd,tf.transpose(xm1_2,perm=(1,0)))
    xxs_x1n = tf.matmul(onesM1d,tf.transpose(xn_2,perm=(1,0)))

    test1_tmp1 = tf.add(xxs_x1n,tf.transpose(xxs_xm1,perm=(1,0)))
    test1_tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxTm1n)

    test1_kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(test1_tmp1,test1_tmp2)))

    # the kernel for the 2nd part test data

    #XXX setup the tf const for the training data

    x = tf.constant(train_x,dtype=tf.float32) #(N,dim)
    x_t2 = tf.constant(test_x2,dtype=tf.float32)

    #XXX formating the kernel of validation data: (M,N)
    xm2_2 = tf.mul(x_t2,x_t2) #(M,d)
    xn_2 = tf.mul(x,x) #(N,d)

    xxTm2n = tf.matmul(x_t2,tf.transpose(x,perm=(1,0))) #(M,N)

    onesM2d = tf.constant(np.ones((M2,dim)),dtype=tf.float32) #(M,d)
    onesNd = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(M,d)
    xxs_xm2 = tf.matmul(onesNd,tf.transpose(xm2_2,perm=(1,0)))
    xxs_x2n = tf.matmul(onesM2d,tf.transpose(xn_2,perm=(1,0)))

    test2_tmp1 = tf.add(xxs_x2n,tf.transpose(xxs_xm2,perm=(1,0)))
    test2_tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxTm2n)

    test2_kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(test2_tmp1,test2_tmp2)))

    test1_prediction = tf.sign(tf.matmul(test1_kernel,betas))
    test2_prediction = tf.sign(tf.matmul(test2_kernel,betas))

saver = tf.train.Saver()

num_steps = 1
with tf.Session() as session:
    saver.restore(session,"klr_model_processed.ckpt")
    for step in range(num_steps):
        if step%1==0:
	    tp1,tp2= session.run([test1_prediction,test2_prediction])

print "result generated"
tp1 =(tp1+1)/2
tp2 =(tp2+1)/2
result_file = open('resultprocessed.txt','w')
for i in range(tp1.shape[0]):
    result_file.write(str(int(tp1[i,0]))+'\n')
for i in range(tp2.shape[0]):
    result_file.write(str(int(tp2[i,0]))+'\n')
result_file.close()
