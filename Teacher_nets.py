import tensorflow as tf
import os.path
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *
    
def encode1(inputs,Training = True,Reuse = False,alpha=0.2, scope_name ='net' ):
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[3, 3 , 512,512 ],tf.float32,tf.glorot_uniform_initializer())
            bias1 = tf.get_variable('bias1',[512],tf.float32,tf.zeros_initializer())
            conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 1, 1, 1], padding='SAME', name='enconv5_1')
            mean, variance = tf.nn.moments(conv1, [0, 1, 2])
            net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)

            weight2 = tf.get_variable('weight2',[3, 3 , 512,512 ],tf.float32,tf.glorot_uniform_initializer())
            bias2 = tf.get_variable('bias2',[512],tf.float32,tf.zeros_initializer())
            conv2 = tf.nn.conv2d(input=net, filter=weight2, strides=[1, 1, 1, 1], padding='SAME', name='enconv5_2')
            mean1, variance1 = tf.nn.moments(conv2, [0, 1, 2])
            net = tf.nn.batch_normalization(conv2, mean1, variance1, bias2, None, 1e-5)
            net = tf.maximum(alpha*net,net)

            weight3 = tf.get_variable('weight3',[3, 3 , 512,512 ],tf.float32,tf.glorot_uniform_initializer())
            bias3 = tf.get_variable('bias3',[512],tf.float32,tf.zeros_initializer())
            conv3 = tf.nn.conv2d(input=net, filter=weight3, strides=[1, 1, 1, 1], padding='SAME', name='enconv5_3')
            mean2, variance2 = tf.nn.moments(conv3, [0, 1, 2])
            net = tf.nn.batch_normalization(conv3, mean2, variance2, bias3, None, 1e-5)
            net = tf.maximum(alpha*net,net)
            net = tf.nn.max_pool(net,[1,2,2,1],[1,2,2,1],padding='SAME')
    return net
    
def encode2(net,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):
    code_shape = net.get_shape().as_list()
    nodes = code_shape[1]*code_shape[2]*code_shape[3]
    net = tf.reshape(net,[code_shape[0],nodes])
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[nodes,4096],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias1',[4096],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(net,weight1)
            mean, variance = tf.nn.moments(net, [0, 1])
            net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)
            if Training: 
                net = tf.nn.dropout(net,0.5)
                tf.add_to_collection('losses1',regularizer(weight1))
    #with tf.variable_scope('encode3',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight2',[4096,4096],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias2',[4096],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(net,weight1)
            mean, variance = tf.nn.moments(net, [0, 1])
            net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)
            if Training: 
                net = tf.nn.dropout(net,0.5)
                tf.add_to_collection('losses2',regularizer(weight1))
    #with tf.variable_scope('encode4',reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight3',[4096,2048],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias3',[2048],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(net,weight1)
            mean, variance = tf.nn.moments(net, [0,1])
            net = tf.nn.batch_normalization(net, mean, variance, bias1, None, 1e-5)
            net = tf.maximum(alpha*net,net)
            if Training: 
                net = tf.nn.dropout(net,0.5)
                tf.add_to_collection('losses3',regularizer(weight1))
    return net
                
def encode3(net,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):                
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
            weight1 = tf.get_variable('weight1',[2048,128],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias1 = tf.get_variable('bias1',[128],tf.float32,initializer=tf.zeros_initializer())
            net = tf.matmul(net,weight1) + bias1
            net = tf.nn.tanh(net)
    return net 













       
       
       
       
       
       
       
       
       
       
       

       
            
