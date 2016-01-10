# accuracy only 0.946120
import numpy as np
import tensorflow as tf
import sys
import tqdm
data = np.matrix(np.genfromtxt('sample_test_x.txt', delimiter=',')[1:,1:])
#truth = np.matrix(np.genfromtxt('truth_train.txt', delimiter=',')[:,1:])

data = data/data.sum(axis=0) # NOTE:normalization
#truth = truth*2-1 # NOTE: shifting the result

data_size = data.shape[0]
d = data.shape[1]

# important settings
layer_size = [d,64,128,64,1]

def weight_size(layer_size):
    sum_ = 0
    for i in range(len(layer_size)-1):
        sum_ = sum_+layer_size[i]*layer_size[i+1]+layer_size[i+1]
    return sum_

print "variable size = ",weight_size(layer_size)
batch_size = 1000
step_num = 1000000
learning_rate = 1e-3

layer_num = len(layer_size)




#TODO : construct batch method


# initialize variables and placeholder
with tf.device('/cpu:0'):
    w_array = []
    b_array = []
    for i in range(layer_num-1):
        w_array.append(tf.Variable(tf.truncated_normal([layer_size[i],layer_size[i+1]], stddev=0.1)))
        b_array.append(tf.Variable(tf.constant(0.1, shape=[layer_size[i+1]])))
    x = tf.placeholder("float", shape=[None, d])
    #y = tf.placeholder("float", shape=[None, 1])
    keep_prob = tf.placeholder("float") #for dropout


# network construction
with tf.device('/cpu:0'):
    h_array = []
    h = x
    for i in range(layer_num-2):
        h = tf.nn.dropout(tf.nn.tanh(tf.matmul(h, w_array[i]) + b_array[i]),keep_prob)
        h_array.append(h)
    out = tf.matmul(h_array[-1],w_array[-1]) + b_array[-1]
    #loss = tf.reduce_sum(tf.square(y-out))#the objective function
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)#the optimizer
    #prediction = tf.equal(tf.sign(out),y)#for testing


#accuracy = tf.reduce_mean(tf.cast(prediction, "float"))

init_op = tf.initialize_all_variables()


saver = tf.train.Saver()
with tf.Session() as sess:
    #sess.run(init_op)
    saver.restore(sess,"DNN64-128-64.ckpt")
    x_batch  = data
    # Training :
    out = sess.run(out,feed_dict={x:x_batch,keep_prob:1.})
    #for i in tqdm.tqdm(range(step_num)):
    #    if i%100 == 0:
    #        acc = sess.run(accuracy,feed_dict={x:x_batch, y: y_batch,keep_prob:1.})
    #        print acc
    #        saver.save(sess,"DNN64-128-64.ckpt")
        # the exact output you're looking for:
    #    x_batch , y_batch = rand_batch(batch_size)
    #    train_step.run(feed_dict={x:x_batch, y: y_batch,keep_prob:0.5})

p = np.sign(out)

print "result = \n",p
result = (p+1)/2
result_file = open('nnresult.txt','w')
for i in range(result.shape[0]):
    result_file.write(str(int(result[i,0]))+'\n')

result_file.close()
