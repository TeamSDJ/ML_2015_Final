# NOTE :
# try to use linear logistic regression to aggregate the KLR models
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

test_data = np.matrix(np.genfromtxt('sample_test_x.txt', delimiter=',')[1:,1:])
test_data = test_data/test_data.sum(axis=0) #NOTE : do the normalization


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
def tfKLRPredictionNoKernel(x_model,xin,betas,gama):
    kernel_ = tfGaussianKernel(x_model,xin,gama)
    return tfKLRPrediction(kernel_,betas)

def tfKLREmbedingNoKernel(x_model,xin,betas,gama):
    kernel_ = tfGaussianKernel(x_model,xin,gama)
    return tf.matmul(kernel_,betas)


with tf.device('/cpu:0'):
    #XXX setup all variables
    betas_array = []
    for i in range(6):
        betas_array.append(tf.Variable(tf.zeros([N, 1],dtype=tf.float32)))
    #XXX kernel placeholder

with tf.device('/cpu:0'):
    #XXX setup the tf const for the training data
    #the input data
    x_model_array = []
    for i in range(6):
        x_model_array.append(tf.constant(data_part(i)[0],dtype=tf.float32)) #(N,dim)
    x_array = x_model_array

    #the validation data
    x_v = tf.constant(data_part(6)[0],dtype=tf.float32)

    embedding_matrix = []


    for i in range(6): #match different model
        embedding_array = []
        for j in range(6): #match different data
            embedding_array.append(tfKLREmbedingNoKernel(x_model_array[i],x_array[j],betas_array[i],gama))
        embedding_result = tf.concat(0, embedding_array)
        embedding_matrix.append(embedding_result)
        embedding_all = tf.concat(1, embedding_matrix)

    #val_prediction_array = []
    #for i in range(6):
    #    val_prediction_array.append(tfKLRPredictionNoKernel(x_model_array[i],x_v,betas_array[i],gama))


saver = tf.train.Saver()
#generate embedding

num_steps = 1
with tf.Session() as session:
    saver.restore(session,"aggklr_model.ckpt")
    for step in range(num_steps):
        if step%1==0:
	        embedding_result= session.run(embedding_all)
	        print "embedding generated ! "


# generate variables for linear aggregation
with tf.device('/cpu:0'):
    weights=tf.Variable(tf.truncated_normal([6,1],stddev=0.001, dtype=tf.float32))

# TODO input all data
#truth[:train_size*6,:]

with tf.device('/cpu:0'):
    embedding = tf.constant(embedding_result,dtype=tf.float32)
    y = tf.nn.tanh(tf.constant(truth[:train_size*6,:],dtype=tf.float32))
    tmp1 = tf.mul(tf.matmul(embedding,weights),y)
    tmp2 = tf.log(tf.add(tf.constant([1],dtype=tf.float32),tf.exp(tf.mul(tf.constant([-1],dtype=tf.float32),tmp1))))
    loss = tf.reduce_sum(tmp2,0)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    prediction = tf.sign(tf.matmul(embedding,weights))

num_steps = 500
with tf.Session() as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        _= session.run(optimizer)
        if step%100==0:
	        p = session.run(prediction)
	        txt = "Ein = "+str(100.*np.sum(p!=truth[:train_size*6,:])/float(train_size*6))# + " Eout = "+str(100*np.sum(vp!=val_y)/val_size)

                print txt
    weights_result = weights.eval()

# teseting the data
with tf.device('/cpu:0'):
    weights_holder=tf.placeholder(tf.float32,shape=(6,1))

# generate the embedding of testing data
with tf.device('/cpu:0'):
    #XXX setup the tf const for the training data
    #the input data
    x_model_array = []
    for i in range(6):
        x_model_array.append(tf.constant(data_part(i)[0],dtype=tf.float32)) #(N,dim)

    #the validation data
    x_t = tf.constant(test_data,dtype=tf.float32)
    embedding_matrix = []
    for i in range(6): #match different model
        embedding_matrix.append(tfKLREmbedingNoKernel(x_model_array[i],x_t,betas_array[i],gama))
    embedding_all = tf.concat(1, embedding_matrix)

    # the prediction
    test_prediction = tf.sign(tf.matmul(embedding_all,weights_holder))

with tf.Session() as session:
    saver.restore(session,"aggklr_model.ckpt")
    for step in range(1):
        if step%1==0:
	        tp = session.run(test_prediction,feed_dict={weights_holder:weights_result})
                print "test predict done! "


tp = (tp+1)/2
result_file = open('liaggresult.txt','w')
for i in range(tp.shape[0]):
    result_file.write(str(int(tp[i,0]))+'\n')

result_file.close()
