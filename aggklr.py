# NOTE :
# - I choose kernel logistic regression since it can be optimized by GD or SGD
# - However, the kernel size is tramendous, so I try to use GPU to accelerate the learning




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
val_size = 15000
data = np.matrix(np.genfromtxt('sample_train_x.txt', delimiter=',')[1:,1:])
truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])

data = data/data.sum(axis=0) # NOTE:normalization
truth = truth*2-1
train_x = data[:train_size,:]
train_y = truth[:train_size,:]


val_x = data[train_size:train_size+val_size,:]
val_y = truth[train_size:train_size+val_size,:]

dim = train_x.shape[1]
N = train_x.shape[0]

#dim = test_x.shape[1]
M = val_x.shape[0]

print "train_size = ",N," val size = ",M
#target = 1+2*np.random.randint(-1,high=1,size = (N,1))

def tfGaussianKernel(xn,xm,gama):
    #XXX formating the kernel of validation data: (M,N)
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
    y = tf.constant(train_y,dtype=tf.float32) #(dim,1)
    x = tf.constant(train_x,dtype=tf.float32) #(N,dim)

    y_v = tf.constant(val_y,dtype=tf.float32)
    x_v = tf.constant(val_x,dtype=tf.float32)
    #XXX formatting the kernel of training data:
    kernel = tfGaussianKernel(x,x,gama)
    val_kernel = tfGaussianKernel(x,x_v,gama)

with tf.device('/cpu:0'):
    #XXX setup the variables
    betas = tf.Variable(tf.zeros([N, 1],dtype=tf.float32))
    #XXX kernel placeholder
    kernel_holder = tf.placeholder(tf.float32,shape=(N,N))
    val_kernel_holder = tf.placeholder(tf.float32,shape=(M,N))

with tf.device('/gpu:0'):
    #the constants of equation
    lemda_const = tf.constant([lemda],dtype=tf.float32)
    #XXX formatting the loss function
    first_term = tf.mul(lemda_const,tf.reduce_sum(tf.mul(betas,tf.matmul(kernel_holder,betas)))) #regulerization
    second_term_tmp = tf.mul(tf.matmul(kernel_holder,betas),y)
    #second_term_tmp = tf.matmul(kernel_holder,betas)
    second_term = tf.log(tf.add(tf.constant([1],dtype=tf.float32),tf.exp(tf.mul(tf.constant([-1],dtype=tf.float32),second_term_tmp))))
    #second_term = tf.reduce_sum(tf.square(tf.sub(y,second_term_tmp)),0)
    loss = tf.add(first_term,tf.reduce_sum(second_term,0))
    #loss = tf.add(first_term,second_term)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    prediction = tf.sign(tf.matmul(kernel_holder,betas))
    val_prediction = tf.sign(tf.matmul(val_kernel_holder,betas))

saver = tf.train.Saver()

with tf.Session() as session:
    kernel_variable = kernel.eval()
    #val_kernel_variable = val_kernel.eval()

print "kernel complete!"
with tf.Session() as session:
    val_kernel_variable = val_kernel.eval()
print "validation kernel complete!"
num_steps = 100
with tf.Session() as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        _= session.run([optimizer], feed_dict={kernel_holder:kernel_variable,val_kernel_holder:val_kernel_variable})
        if step%1==0:

	    _,p,vp= session.run([optimizer,prediction,val_prediction], feed_dict={kernel_holder:kernel_variable,val_kernel_holder:val_kernel_variable})
	    txt = "Ein = "+str(100*np.sum(p!=train_y)/train_size) + " Eout = "+str(100*np.sum(vp!=val_y)/val_size)
            print txt
    saver.save(session,"klr_model.ckpt")
