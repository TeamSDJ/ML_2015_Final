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

# NOTE : set the training_size and test_size to 15000, so the the GPU memory can handle
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

test_data = np.matrix(np.genfromtxt('sample_test_x.txt', delimiter=',')[1:,1:])
test_data = test_data/test_data.sum(axis=0) #NOTE : do the normalization


test_x = test_data
#val_y = data_part(6)[1]

dim = test_x.shape[1]
M = test_x.shape[0]
N = train_size

print "train_size = ",N," test size = ",M
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
    #XXX setup the tf const for the training data

    x_array = []
    for i in range(6):

        x_array.append(tf.constant(data_part(i)[0],dtype=tf.float32)) #(N,dim)

    x_t = tf.constant(test_x,dtype=tf.float32)
    #XXX formatting the kernel of training data & validation data:
    test_kernel_array = []
    for i in range(6):
        #kernel_array.append(tfGaussianKernel(x_array[i],x_array[i],gama))
        test_kernel_array.append(tfGaussianKernel(x_array[i],x_t,gama))

with tf.device('/cpu:0'):
    #XXX setup all variables
    betas_array = []
    for i in range(6):
        betas_array.append(tf.Variable(tf.zeros([N, 1],dtype=tf.float32)))
    #XXX kernel placeholder
    #kernel_holder_array = []
    test_kernel_holder_array = []
    for i in range(6):
        test_kernel_holder_array.append(tf.placeholder(tf.float32,shape=(M,N)))


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

with tf.device('/cpu:0'):
    #the constants of equation
    #loss_array = []
    #optimizer_array = []
    #prediction_array = []
    test_prediction_array = []
    #for i in range(6):
    #    loss_array.append(tfKLRLoss(y_array[i],betas_array[i],kernel_holder_array[i],lemda))
    #for i in range(6):
    #    optimizer_array.append(tf.train.AdamOptimizer(0.001).minimize(loss_array[i]))

    for i in range(6):
        #prediction_array.append(tfKLRPrediction(kernel_holder_array[i],betas_array[i]))
        test_prediction_array.append(tfKLRPrediction(test_kernel_holder_array[i],betas_array[i]))


saver = tf.train.Saver()


with tf.Session() as session:
    test_kernel_variable_array = []
    for i in range(6):
        test_kernel_variable_array.append(test_kernel_array[i].eval())

print "testidation kernel operation complete!"
num_steps = 1
result_array = []
with tf.Session() as session:
    saver.restore(session,"aggklr_model.ckpt")
    for i in range(6):
        print "part ",i
        for step in range(num_steps):
            #_= session.run([optimizer_array[i]], feed_dict={kernel_holder_array[i]:kernel_variable_array[i],test_kernel_holder_array[i]:test_kernel_variable_array[i]})
	        tp= session.run(test_prediction_array[i], feed_dict={test_kernel_holder_array[i]:test_kernel_variable_array[i]})
        result_array.append(np.matrix(tp))
        print tp,type(tp)


result = np.sign(np.sum(np.concatenate(result_array,axis=1),axis=1))
print "result = \n",result
result = (result+1)/2
result_file = open('aggresult.txt','w')
for i in range(result.shape[0]):
    result_file.write(str(int(result[i,0]))+'\n')

result_file.close()

# TODO do some shattering
