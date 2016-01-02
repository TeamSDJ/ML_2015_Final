# NOTE :
# - I choose kernel logistic regression since it can be optimized by GD or SGD
# - However, the kernel size is tramendous, so I try to use GPU to accelerate the learning
# TODO: try to split the training data into 6 peaces



import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt
#XXX read in the files

# NOTE : the model const lemda and kernel const are choosen by try-and-error
# for larger gama , we can get smaller Eval, however, for too large gamm, Eval increase
# the lemda should be set low, so that the learning can be faster. (Larger panelty on cross-entropy)
lemda = 0.5
gama = 1000000000

# NOTE : set the training_size and val_size to 15000, so the the GPU memory can handle
train_size = 15000

peace_num = 6
data = np.matrix(np.genfromtxt('sample_train_x.txt', delimiter=',')[1:,1:])
truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])

# NOTE:normalization and shift the target to +1 -1
data = data/data.sum(axis=0)
truth = truth*2-1

def data_part(part,N = train_size): # 0~5 training set # 6th part is the validation set
    train_x = data[part*N:part*N+N,:]
    train_y = truth[part*N:part*N+N,:]
    return train_x,train_y

val_x = data_part(6)[0]
val_y = data_part(6)[1]

dim = val_x.shape[1]
M = val_x.shape[0]
N = train_size

print "train_size = ",N," val size = ",M
#target = 1+2*np.random.randint(-1,high=1,size = (N,1))

def tfGaussianKernel(xn,xm,gama):
    #XXX formating the kernel of validation data: (M,N)
    M = xm._shape[0]
    N = xn._shape[0]
    xm_2 = tf.mul(xm,xm) #(M,d)
    xn_2 = tf.mul(xn,xn) #(N,d)
    xxTmn = tf.matmul(xm,tf.transpose(xn,perm=(1,0))) #(M,N)
    onesMd = tf.constant(np.ones((M,dim)),dtype=tf.float32) #(M,d)
    onesNd = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(M,d)
    xxs_xm = tf.matmul(onesNd,tf.transpose(xm_2,perm=(1,0)))
    xxs_xn = tf.matmul(onesMd,tf.transpose(xn_2,perm=(1,0)))
    tmp1 = tf.add(xxs_xn,tf.transpose(xxs_xm,perm=(1,0)))
    tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxTmn)
    kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(tmp1,tmp2)))
    return kernel

with tf.device('/gpu:0'):
    #XXX setup the tf const for the training data
    y_array = []
    x_array = []
    for i in range(6):
        y_array.append(tf.constant(data_part(i)[1],dtype=tf.float32)) #(dim,1)
        x_array.append(tf.constant(data_part(i)[0],dtype=tf.float32)) #(N,dim)

    y_v = tf.constant(data_part(6)[1],dtype=tf.float32)
    x_v = tf.constant(data_part(6)[0],dtype=tf.float32)
    #XXX formatting the kernel of training data & validation data:
    kernel_array = []
    val_kernel_array = []
    for i in range(6):
        kernel_array.append(tfGaussianKernel(x_array[i],x_array[i],gama))
        val_kernel_array.append(tfGaussianKernel(x_array[i],x_v,gama))

with tf.device('/cpu:0'):
    #XXX setup all variables
    betas_array = []
    for i in range(6):
        betas_array.append(tf.Variable(tf.zeros([N, 1],dtype=tf.float32)))
    #XXX kernel placeholder
    kernel_holder_array = []
    val_kernel_holder_array = []
    for i in range(6):
        kernel_holder_array.append(tf.placeholder(tf.float32,shape=(N,N)))
        val_kernel_holder_array.append(tf.placeholder(tf.float32,shape=(M,N)))


def tfKLRLoss(y,betas,kernel_holder,lemda):
    lemda_const = tf.constant([lemda],dtype=tf.float32)
    #XXX formatting the loss function
    first_term = tf.mul(lemda_const,tf.reduce_sum(tf.mul(betas,tf.matmul(kernel_holder,betas)))) #regulerization
    second_term_tmp = tf.mul(tf.matmul(kernel_holder,betas),y)
    #second_term_tmp = tf.matmul(kernel_holder,betas)
    second_term = tf.log(tf.add(tf.constant([1],dtype=tf.float32),tf.exp(tf.mul(tf.constant([-1],dtype=tf.float32),second_term_tmp))))
    #second_term = tf.reduce_sum(tf.square(tf.sub(y,second_term_tmp)),0)
    loss = tf.add(first_term,tf.reduce_sum(second_term,0))
    return loss

def tfKLRPrediction(kernel_holder,betas):
    prediction = tf.sign(tf.matmul(kernel_holder,betas))
    return prediction

with tf.device('/gpu:0'):
    #the constants of equation
    loss_array = []
    optimizer_array = []
    prediction_array = []
    val_prediction_array = []
    for i in range(6):
        loss_array.append(tfKLRLoss(y_array[i],betas_array[i],kernel_holder_array[i],lemda))
    for i in range(6):
        optimizer_array.append(tf.train.AdamOptimizer(0.001).minimize(loss_array[i]))

    for i in range(6):
        prediction_array.append(tfKLRPrediction(kernel_holder_array[i],betas_array[i]))
        val_prediction_array.append(tfKLRPrediction(val_kernel_holder_array[i],betas_array[i]))


saver = tf.train.Saver()

with tf.Session() as session:
    kernel_variable_array = []
    for i in range(6):
        kernel_variable_array.append(kernel_array[i].eval())
    #val_kernel_variable = val_kernel.eval()

print "kernel operation complete!"

with tf.Session() as session:
    val_kernel_variable_array = []
    for i in range(6):
        val_kernel_variable_array.append(val_kernel_array[i].eval())

print "validation kernel operation complete!"
num_steps = 20
with tf.Session() as session:
    tf.initialize_all_variables().run()
    for i in range(6):
        print "part ",i
        for step in range(num_steps):
            _= session.run([optimizer_array[i]], feed_dict={kernel_holder_array[i]:kernel_variable_array[i],val_kernel_holder_array[i]:val_kernel_variable_array[i]})
            if step%1==0:
    	        p,vp= session.run([prediction_array[i],val_prediction_array[i]], feed_dict={kernel_holder_array[i]:kernel_variable_array[i],val_kernel_holder_array[i]:val_kernel_variable_array[i]})
    	        txt = " Ein = "+str(float(100*np.sum(p!=data_part(i)[1]))/float(N))+" Eout = "+str(float(100*np.sum(vp!=val_y))/float(M))
                print txt
        saver.save(session,"aggklr_model.ckpt")
