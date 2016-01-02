# kernel logistic regression can be optimized by GD or SGD



import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt
#XXX read in the files

#XXX First reload the data we generated in 1_notmist.ipynb.
#trainfile = open('train.dat.txt')
#testfile = open('test.dat.txt')

#train_data = [[float(element) for element in line.split()] for line in trainfile]
#test_data = [[float(element) for element in line.split()] for line in testfile]

data = np.genfromtxt('sample_train_x.txt', delimiter=',')
truth = np.genfromtxt('truth_train.txt', delimiter=',')



train_x = np.matrix(data[1:10001,1:])
train_y = np.matrix(truth[:10000,1:])*2-1

train_x = train_x/train_x.sum(axis = 0)


np.shape(train_x)
np.shape(train_y)
#train_x = np.matrix(train_data)[:,:2]
#train_y = np.matrix(train_data)[:,2:]

#test_x = np.matrix(test_data)[:,:2]
#test_y = np.matrix(test_data)[:,2:]


dim = train_x.shape[1]
N = train_x.shape[0]

#dim = test_x.shape[1]
#M = test_x.shape[0]

lemda = 5
gama = 100000



#target = 1+2*np.random.randint(-1,high=1,size = (N,1))


with tf.device('/gpu:0'):
    #XXX setup the tf const for the training data
    y = tf.constant(train_y,dtype=tf.float32) #(dim,1)
    x = tf.constant(train_x,dtype=tf.float32) #(N,dim)

    #y_t = tf.constant(test_y,dtype=tf.float32)
    #x_t = tf.constant(test_x,dtype=tf.float32)
    #XXX formatting the kernel of training data:
    x_2 = tf.mul(x,x) #(N,d)
    xxT = tf.matmul(x,tf.transpose(x,perm=(1,0))) #(N,N)
    ones = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(N,d)
    xxTs = tf.matmul(ones,tf.transpose(x_2,perm=(1,0)))
    tmp1 = tf.add(xxTs,tf.transpose(xxTs,perm=(1,0)))
    tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxT)
    kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(tmp1,tmp2)))


#XXX formating the kernel of testing data: (M,N)
#xm_2 = tf.mul(x_t,x_t) #(M,d)
#xn_2 = tf.mul(x,x) #(N,d)

#xxTmn = tf.matmul(x_t,tf.transpose(x)) #(M,N)

#onesMd = tf.constant(np.ones((M,dim)),dtype=tf.float32) #(M,d)
#onesNd = tf.constant(np.ones((N,dim)),dtype=tf.float32) #(M,d)
#xxs_xm = tf.matmul(onesNd,tf.transpose(xm_2))
#xxs_xn = tf.matmul(onesMd,tf.transpose(xn_2))

#test_tmp1 = tf.add(xxs_xn,tf.transpose(xxs_xm))
#test_tmp2 = tf.mul(tf.constant([-2.],dtype=tf.float32),xxTmn)

#test_kernel = tf.exp(tf.mul(tf.constant([-gama],dtype=tf.float32),tf.add(test_tmp1,test_tmp2)))


with tf.device('/cpu:0'):
    #XXX setup the variables
    betas = tf.Variable(tf.zeros([N, 1],dtype=tf.float32))
    #betasT = tf.transpose(betas,perm=(1,0))
    #XXX kernel placeholder
    kernel_holder = tf.placeholder(tf.float32,shape=(N,N))
    #test_kernel_holder = tf.placeholder(tf.float32,shape=(M,N))

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
    optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)
    prediction = tf.sign(tf.matmul(kernel_holder,betas))
#test_prediction = tf.sign(tf.matmul(test_kernel_holder,betas))

with tf.Session() as session:
    kernel_variable = kernel.eval()


print "kernel complete!"
#with tf.Session() as session:
#    test_kernel_variable = test_kernel.eval()

#np.shape(test_kernel_variable)
num_steps = 10000
with tf.Session() as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        _= session.run([optimizer], feed_dict={kernel_holder:kernel_variable})
        if step%10==0:
	    #p=prediction.eval(feed_dict={kernel_holder:kernel_variable})
            #print type(p)
	    _,p= session.run([optimizer,prediction], feed_dict={kernel_holder:kernel_variable})
	    txt = " Ein = "+str(100*np.sum(p!=train_y)/float(p.shape[0]))#+" Eout = "+str(100*np.sum(tp!=test_y)/float(tp.shape[0]))
            print txt
            #time.sleep(0.5)
            #print '\r','loss = ',l,'accuracy = ',100*np.sum(p==target)/float(p.shape[0])
