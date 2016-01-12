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
gama = 1e9
learning_rate = [0.01,0.001,0.001,0.001,0.001,0.001]
# NOTE : set the training_size and val_size to 15000, so the the GPU memory can handle
train_size = 15000

peace_num = 6
data = np.matrix(np.genfromtxt('train_x.txt', delimiter=',')[1:,1:])
truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])

np.shape(data)
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

print "train_size = ",N," val size = ",M,"dim = ",dim
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

with tf.device('/cpu:0'):
    #XXX setup all variables
    betas_array = []
    for i in range(6):
        betas_array.append(tf.Variable(tf.zeros([N, 1],dtype=tf.float32)))
    weights = tf.Variable(tf.truncated_normal([6,1], stddev=0.1))
    x_train = tf.placeholder(tf.float32,shape=(train_size,dim))
    y_train = tf.placeholder(tf.float32,shape=(train_size,1))
    #XXX kernel placeholder
    #kernel_holder_array = []
    #val_kernel_holder_array = []
    #for i in range(6):
    #    kernel_holder_array.append(tf.placeholder(tf.float32,shape=(N,N)))
    #    val_kernel_holder_array.append(tf.placeholder(tf.float32,shape=(M,N)))


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

def tfLRLoss(y,out):
    #XXX formatting the loss function
    second_term_tmp = tf.mul(out,y)
    #second_term_tmp = tf.matmul(kernel_holder,betas)
    second_term = tf.log(tf.add(tf.constant([1],dtype=tf.float32),tf.exp(tf.mul(tf.constant([-1],dtype=tf.float32),second_term_tmp))))
    #second_term = tf.reduce_sum(tf.square(tf.sub(y,second_term_tmp)),0)
    loss = tf.reduce_sum(second_term,0)
    return loss

def tfKLROut(kernel_holder,betas):
    return tf.matmul(kernel_holder,betas)

def tfKLRPrediction(kernel_holder,betas):
    prediction = tf.sign(tf.matmul(kernel_holder,betas))
    return prediction

def tfError(y,out):
    #XXX formatting the loss function
    second_term_tmp = tf.mul(out,y)
    #second_term_tmp = tf.matmul(kernel_holder,betas)
    second_term = tf.log(tf.add(tf.constant([1],dtype=tf.float32),tf.exp(tf.mul(tf.constant([-1],dtype=tf.float32),second_term_tmp))))
    #second_term = tf.reduce_sum(tf.square(tf.sub(y,second_term_tmp)),0)
    return second_term


with tf.device('/cpu:0'):
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
        kernel_array.append(tfGaussianKernel(x_array[i],x_train,gama))
        val_kernel_array.append(tfGaussianKernel(x_array[i],x_v,gama))
    #the constants of equation
    out_array = []
    for i in range(6):
        out_array.append(tfKLROut(kernel_array[i],betas_array[i]))
    out = tf.concat(1,out_array)
    out._shape
    out_final = tf.matmul(out,weights)
    loss = tfLRLoss(y_train,out_final)
    #second_term_tmp = tf.matmul(kernel_holder,betas)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    val_out_array = []
    for i in range(6):
        val_out_array.append(tfKLROut(val_kernel_array[i],betas_array[i]))
    val_out = tf.concat(1,val_out_array)
    val_out_final = tf.matmul(val_out,weights)
    prediction = tf.sign(out_final)
    val_prediction = tf.sign(val_out_final)
    Ein = tfError(y_train,out_final)
    Eval = tfError(y_v,val_out_final)

Ein_ = tf.reduce_mean(Ein)
Eval_ = tf.reduce_mean(Eval)

saver = tf.train.Saver()


num_steps = 600
with tf.Session() as session:
    tf.initialize_all_variables().run()


    for step in range(num_steps):
        x,y = data_part(step%6,train_size)
        _= session.run(optimizer,feed_dict={x_train:x,y_train:y})
        if step%1==0:
	        p,vp,Ein,Eval= session.run([prediction,val_prediction,Ein_,Eval_,feed_dict={x_train:x,y_train:y})
	        txt = " Ein0,1 = "+str(float(100*np.sum(np.matrix(p)!=data_part(step%6,train_size)[1]))/float(N))+" Eval0,1 = "+str(float(100*np.sum(np.matrix(vp)!=val_y))/float(M))
            print txt
            print "Einp=",Ein,"Evalp=",Eval
    saver.save(session,"aggklrnetwork_model.ckpt")


# TODO do some shattering
