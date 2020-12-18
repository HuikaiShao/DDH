import tensorflow as tf
import os.path
import glob
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *

regularizer = tf.contrib.layers.l2_regularizer(0.0005)
def encode1(inputs,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[3, 3 , 3, 16],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[16],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=inputs, filter=weight1, strides=[1, 4, 4, 1], padding='VALID', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net,net)             
        net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
        return net

def encode2(net,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):            
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[5, 5, 16, 32],tf.float32,tf.glorot_uniform_initializer())
        bias1 = tf.get_variable('bias1',[32],tf.float32,tf.zeros_initializer())
        conv1 = tf.nn.conv2d(input=net, filter=weight1, strides=[1, 2, 2, 1], padding='SAME', name='deconv1')
        mean, variance = tf.nn.moments(conv1, [0, 1, 2])
        net = tf.nn.batch_normalization(conv1, mean, variance, bias1, None, 1e-5)
        net = tf.maximum(alpha*net,net)             
        net = tf.nn.max_pool(net,[1,2,2,1],[1,1,1,1],'VALID')
    return net
   
def encode3(inputs,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):   
    code_shape = inputs.get_shape().as_list()
    nodes = code_shape[1]*code_shape[2]*code_shape[3]
    inputs = tf.reshape(inputs,[code_shape[0],nodes])
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[nodes,512],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[512],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(inputs,weight1)
        net = net + bias1
        net = tf.maximum(alpha*net,net)
        if Training: net = tf.nn.dropout(net,0.5)
        tf.add_to_collection('losses',regularizer(weight1)) 
    return net

def encode4(net,Training = True,Reuse = False,alpha=0.2, scope_name ='net'):        
    with tf.variable_scope(scope_name,reuse = Reuse) as scope:
        weight1 = tf.get_variable('weight1',[512,128],tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias1 = tf.get_variable('bias1',[128],tf.float32,initializer=tf.zeros_initializer())
        net = tf.matmul(net,weight1) + bias1
        net = tf.nn.tanh(net)
    return net  

