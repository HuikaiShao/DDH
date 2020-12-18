import glob
import os.path
import random
import numpy as np
import tensorflow as tf
import Teacher_nets as teacher_net
import Student_nets as student_net
import vgg16
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
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
def Hard_loss(T_code2,S_code2,label_batch,batch_size,num_per):

    T_code_matrix = tf.matmul(T_code2,tf.transpose(T_code2))
    label_matrix = tf.matmul(label_batch,tf.transpose(label_batch))
    T_code_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(T_code_matrix),[batch_size]),[batch_size,batch_size]))
    T_all_distance = T_code_diag + tf.transpose(T_code_diag) - 2*T_code_matrix
    T_positive = label_matrix * T_all_distance
    T_negative = (1-label_matrix) * T_all_distance

    S_code_matrix = tf.matmul(S_code2,tf.transpose(S_code2))
    S_code_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(S_code_matrix),[batch_size]),[batch_size,batch_size]))
    S_all_distance = S_code_diag + tf.transpose(S_code_diag) - 2*S_code_matrix
    S_positive = label_matrix * S_all_distance
    S_negative = (1-label_matrix) * S_all_distance

    positive_loss = tf.Variable(tf.zeros([1]))
    negative_loss = tf.Variable(tf.zeros([1]))
    for i in range(int(batch_size/num_per)):
        T_sub_positive = T_positive[i*num_per:(i+1)*num_per,i*num_per:(i+1)*num_per]
        S_sub_positive = S_positive[i*num_per:(i+1)*num_per,i*num_per:(i+1)*num_per]
        positive_loss = tf.concat([positive_loss,[tf.reduce_max(S_sub_positive)-tf.reduce_min(T_sub_positive)]], axis=0) 
        positive_loss = positive_loss[1:]
        T_n = tf.concat([T_negative[i*num_per:(i+1)*num_per,:i*num_per],\
                            T_negative[i*num_per:(i+1)*num_per,(i+1)*num_per:]], axis=1)
        S_n = tf.concat([S_negative[i*num_per:(i+1)*num_per,:i*num_per],\
                            S_negative[i*num_per:(i+1)*num_per,(i+1)*num_per:]], axis=1)
        negative_loss = tf.concat([negative_loss,tf.reduce_max(T_n)-[tf.reduce_min(S_n)]], axis=0) 
        negative_loss = negative_loss[1:]
    
    loss = tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)    
    return  loss

#-----------------------------------------------------------------------------------------------------------------------
batch_size = 50
capacity=1000+3*batch_size
omega_size = 30
softmax_temperature=5
_eps=1e-8
def main():
    tf.reset_default_graph()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"       
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    logs_train_dir = './model/save_DDH/model.ckpt'
    path_img = 'SF.txt' # the list of images
    num_per = 10 # the number of images per category 
    train_data,train_lable = read_image(path_img, num_per)
    train_size=len(train_data)
    image_batch,label_batch = get_batch(train_data, train_lable,train_size,batch_size,capacity,False)

    global_step = tf.Variable(0,trainable = False)
    opt = tf.train.RMSPropOptimizer(0.0001,0.9)
    # student   
    feature_s1 = student_net.encode1(image_batch, False, False, scope_name = 'student1')
    feature_s2 = student_net.encode2(feature_s1, False, False, scope_name = 'student2')
    feature_s3 = student_net.encode3(feature_s2, False, False, scope_name = 'student3')
    feature_s4 = student_net.encode4(feature_s3, False, False, scope_name = 'student4')
    code_s = tf.sign(feature_s4)   

    hash_loss_s, student_archer_distance, student_sabor_distance = Hash_loss(feature_s4,label_batch,batch_size,omega_size)
    q_loss_s = tf.reduce_mean(tf.pow(tf.subtract(feature_s4, code_s), 2.0))
    DHN_loss = hash_loss_s + 0.5 * q_loss_s
    # teacher
    vgg = vgg16.Vgg16()
    vgg.build(image_batch)
    feature_t = vgg.pool4
    feature_t1 = teacher_net.encode1(feature_t, False, False, scope_name = 'teacher1')
    feature_t2 = teacher_net.encode2(feature_t1, False, False, scope_name = 'teacher2')
    feature_t3 = teacher_net.encode3(feature_t2, False, False, scope_name = 'teacher3')
    code_t = tf.sign(feature_t3)
    
    _, teacher_archer_distance, teacher_sabor_distance = Hash_loss(feature_t3,label_batch,batch_size,omega_size)
    # loss
    rela_loss = tf.reduce_mean(tf.pow(tf.subtract(student_archer_distance, teacher_archer_distance), 2.0))+\
                    tf.reduce_mean(tf.pow(tf.subtract(student_sabor_distance, teacher_sabor_distance), 2.0))

    hard_loss = Hard_loss(feature_t3, feature_s4, label_batch, batch_size, num_per)

    loss = rela_loss + hard_loss + DHN_loss

    all_vars = tf.trainable_variables()
    t_vars = [var for var in all_vars if 'teacher' in var.name]
    s_vars = [var for var in all_vars if 'student' in var.name] 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = opt.minimize(loss, global_step = global_step, var_list = s_vars)  

    saver_teacher= tf.train.Saver(t_vars)

    sess.run(tf.global_variables_initializer())
    saver_teacher.restore(sess,"./model/save_teacher/model.ckpt-100") 
    saver = tf.train.Saver(s_vars, max_to_keep=0)
    #saver.restore(sess,"./saver_DDH/model.ckpt-100")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('start train_bottle')
    steps=int(len(train_data)/batch_size)
    epoch=680
    count = 0000
    try:
        for e in range(epoch):
            if coord.should_stop():
                break
            for step in range(steps):
                count=count+1               
                _, loss_ = sess.run([optimizer, loss])
                if( count%10 == 0):                    
                    print("After %d apoch %d training step(s),the loss is %g." % (e+1,count,loss_))
                if( count%100 == 0):
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

    image_batch,label_batch= tf.train.batch([image,label],batch_size= batch_size,num_threads= 1, capacity = Capacity)
    
    label_batch = tf.cast(label_batch, tf.float32)
    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
