import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import Student_nets as student_net
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
import os

#-----------------------------------------------------------------------------------------------------------------------
def read_image(path, num_per):
    train_data = []
    train_lable = []

    train_datas = []
    train_lables = []
    lables=[]

    roi = open(path)  
    roi_path = roi.readlines()
    classnum = len(roi_path)//num_per
    trainnum = 0
    testnum = 0
    i = 0
    for i,image_list in enumerate(roi_path):
        if i % num_per < num_per/2:
            train_datas.append(image_list[:-1])
            train_lables.append(int(i/num_per))
            trainnum = trainnum +1
    lables=train_lables
    train_lable = np.zeros([trainnum, classnum], np.int64)

    i = 0
    for label in train_lables:
        train_lable[i][label] = 1
        i = i + 1

    print(trainnum,classnum)
    train_lables = train_lable.reshape([trainnum*classnum])

    return train_datas,train_lables
#-----------------------------------------------------------------------------------------------------------------------
def Hash_loss(feature,label_batch,batch_size,omega_size):
    archer_feature,sabor_feature = tf.split(feature,[omega_size,batch_size-omega_size],axis = 0)
    archer_label,sabor_label = tf.split(label_batch,[omega_size,batch_size-omega_size],axis = 0)
    archer_matrix = tf.matmul(archer_feature,tf.transpose(archer_feature))
    sabor_matrix = tf.matmul(sabor_feature,tf.transpose(sabor_feature))

    archer_Similarity = tf.matmul(archer_label,tf.transpose(archer_label))
    sabor_Similarity = tf.matmul(archer_label,tf.transpose(sabor_label))
    archer_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[omega_size]),[omega_size,omega_size]))
    archer_sabor_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[batch_size-omega_size]),[batch_size-omega_size,omega_size]))
    sabor_diag = tf.reshape(tf.tile(tf.diag_part(sabor_matrix),[omega_size]),[omega_size,batch_size-omega_size])

    archer_distance = archer_diag + tf.transpose(archer_diag) - 2*archer_matrix
    sabor_distance = sabor_diag + archer_sabor_diag - 2*tf.matmul(archer_feature,tf.transpose(sabor_feature))
    archer_loss = tf.reduce_mean(1/2*archer_Similarity*archer_distance + 1/2*(1-archer_Similarity)*tf.maximum(180-archer_distance,0))
    sabor_loss = tf.reduce_mean(1/2*sabor_Similarity*sabor_distance + 1/2*(1-sabor_Similarity)*tf.maximum(180-sabor_distance,0))
    hash_loss = archer_loss + sabor_loss

    return hash_loss,archer_distance,sabor_distance

#-----------------------------------------------------------------------------------------------------------------------
batch_size = 20
omega_size = 10
capacity=1000+3*batch_size

def main():
    tf.reset_default_graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"           
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    logs_train_dir = './save_student/model.ckpt'
    path = 'SF.txt'
    num_per = 10
    train_data,train_label = read_image(path, num_per)
    train_size=len(train_data)
    batch, label = get_batch(train_data, train_label,train_size,batch_size,capacity,True)
    
    global_step = tf.Variable(0,trainable = False,name = "global_step")
    opt = tf.train.RMSPropOptimizer(0.001,0.9)

    feature_s1 = student_net.encode1(batch, False, False, scope_name = 'student1')
    feature_s2 = student_net.encode2(feature_s1, False, False, scope_name = 'student2')
    feature_s3 = student_net.encode3(feature_s2, False, False, scope_name = 'student3')
    feature_s4 = student_net.encode4(feature_s3, False, False, scope_name = 'student4')
    code = tf.sign(feature_s4)

    hash_loss, _, _ = Hash_loss(feature_s4, label, batch_size, omega_size)
    q_loss = tf.reduce_mean(tf.pow(tf.subtract(feature_s4, code), 2.0))
    DHN_loss = hash_loss+ 0.5*q_loss
 
    all_vars = tf.trainable_variables()
    t_vars = [var for var in all_vars if 'student' in var.name]
        
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(DHN_loss,global_step = global_step, var_list = t_vars)

    sess.run(tf.global_variables_initializer())      
    saver = tf.train.Saver(t_vars,max_to_keep=0)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    steps=int(len(train_data)/batch_size)
    epoch=500
    count=0000
    print('start train_bottle')
    try:
        for e in range(epoch):
            if coord.should_stop():
                break
            for step in range(steps): 
                count=count+1
                _, loss_= sess.run([optimizer,DHN_loss])
                if( (count)%10 == 0):
                    print("After %d epoch %d training step(s),the loss is %g." % (e, count, loss_))
                if( (count)%10000 == 0):
                    saver.save(sess,logs_train_dir,global_step=count)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

#-----------------------------------------------------------------------------------------------------------------------
def get_batch(image, label,label_size,batch_size, Capacity,Shuffle):
    
    classnum=len(label)//len(image)
    image = tf.cast(image, tf.string)
    label = tf.convert_to_tensor(label,tf.int64)
    label = tf.reshape(label,[label_size,classnum])
    
    input_queue = tf.train.slice_input_producer([image, label],shuffle = Shuffle,capacity = Capacity)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [224, 224])
  
    image_batch,label_batch = tf.train.batch([image,label],batch_size= batch_size,num_threads= 1, capacity = Capacity)
    
    label_batch = tf.cast(label_batch, tf.float32)
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
