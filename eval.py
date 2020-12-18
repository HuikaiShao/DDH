import glob
import random
import numpy as np
import tensorflow as tf
import vgg16
import Teacher_nets as teacher_net
import Student_nets as student_net
from tensorflow.python.platform import gfile
from tensorflow.python.ops import array_ops
import os

def read_image(path):
    tatol_datas = []
    roi = open(path)
    roi_path = roi.readlines()
    total_data = []
    i = 0
    for i,image_list in enumerate(roi_path):
            tatol_datas.append(image_list[:-1])
    return tatol_datas

def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"       
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    path = 'SF.txt'
    num_per = 10
    model_path = "./model/saver_DDH/model.ckpt-100"
    total_data = read_image(path)
    print(len(total_data))
    total_image = get_batch(total_data,1,2040,False)
    Flag = 'Student'
    if Flag == 'Teacher':
        vgg = vgg16.Vgg16()
        vgg.build(total_image)
        feature_t = vgg.pool4
        feature_t1 = teacher_net.encode1(feature_t, False, False, scope_name = 'teacher1')
        feature_t2 = teacher_net.encode2(feature_t1, False, False, scope_name = 'teacher2')
        feature = teacher_net.encode3(feature_t2, False, False, scope_name = 'teacher3')
    elif Flag == 'Student':
        feature_s1 = student_net.encode1(total_image, False, False, scope_name = 'student1')
        feature_s2 = student_net.encode2(feature_s1, False, False, scope_name = 'student2')
        feature_s3 = student_net.encode3(feature_s2, False, False, scope_name = 'student3')
        feature = student_net.encode4(feature_s3, False, False, scope_name = 'student5')
        
    all_vars = tf.trainable_variables()
    if Flag == 'Teacher':
        vars_load = [var for var in all_vars if 'teacher' in var.name]
    elif Flag == 'Student':
        vars_load = [var for var in all_vars if 'student' in var.name]   
    print(vars_load)
    saver = tf.train.Saver(vars_load)  
    saver.restore(sess, model_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    
    true_list = []
    false_list=[]   
    result_code = []
    try:
        for i in range(len(total_data)):
            if coord.should_stop():
                break
            code = sess.run(feature)
            code = np.reshape(code,[1,128])
            code = np.sign(code)
            result_code.append(code)
        print('extract features')
        result_code=np.reshape(result_code,[len(total_data),128])
        
        for i in range(len(result_code)):            
            if(i>=(int(i/num_per)*num_per + num_per/2) and i<(int(i/num_per)*num_per+num_per)):
                for j in range(i+1,len(result_code)):                
                    if(j>=(int(i/num_per)*num_per + num_per/2) and j<(int(i/num_per)*num_per+num_per)):
                        true_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
                    else: 
                        false_list.append(np.sqrt(np.sum(np.square(result_code[i] - result_code[j]))))
            print(i)
        np.savetxt("./true.txt",true_list,fmt="%f")
        np.savetxt("./false.txt",false_list,fmt="%f")
        print('Done')
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
  
                    
def get_batch(image,batch_size, Capacity,Shuffle):
        
    image = tf.cast(image, tf.string)   
    input_queue = tf.train.slice_input_producer([image],shuffle = Shuffle,capacity = Capacity)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [224, 224])
    
    image_batch = tf.train.batch([image],batch_size= batch_size,num_threads= 1, 
                                                capacity = Capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch


if __name__ == '__main__':
    main()
