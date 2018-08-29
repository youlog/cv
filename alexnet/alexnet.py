# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.io as scio

### 读取TFRecord文件
### 使用tf.data.Dataset

# 定义数据解析函数，解析TFRecord文件的方法
def parser(record):
    # 解析读取的样例；解析多个样例用parse_example()函数
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    # 从二进制解析出图像的像素矩阵，并还原图像
    decoded_image = tf.decode_raw(features['image'], tf.uint8)
    decoded_image = tf.reshape(decoded_image, tf.cast([features['height'], features['width'], features['channel']], tf.int32))
    decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)
    label = features['label']
    return decoded_image, label

# 色彩调整函数，被preprocess()调用
def color_adjust(image, op_type=None):
    if op_type == 1:
        image = tf.image.random_brightness(image, 0.5)
        image = tf.image.random_contrast(image, 0.3, 3.0)
        image = tf.image.random_hue(image, 0.5)
        image = tf.image.random_saturation(image, 0, 5.0)
    elif op_type == 2:
        image = tf.image.random_contrast(image, 0.3, 3.0)
        image = tf.image.random_brightness(image, 0.5)
        image = tf.image.random_saturation(image, 0, 5.0)
        image = tf.image.random_hue(image, 0.5)
    elif op_type == 3:
        image = tf.image.random_saturation(image, 0, 5.0)
        image = tf.image.random_hue(image, 0.5)
        image = tf.image.random_brightness(image, 0.5)
        image = tf.image.random_contrast(image, 0.3, 3.0)
    elif op_type == 4:
        image = tf.image.random_hue(image, 0.5)
        image = tf.image.random_saturation(image, 0, 5.0)
        image = tf.image.random_contrast(image, 0.3, 3.0)
        image = tf.image.random_brightness(image, 0.5)
    else:
        return image
    return tf.clip_by_value(image, 0.0, 1.0)

# 图像预处理函数
def preprocess(image, image_size, bbox=None):
    import random
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 翻转
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    # 对角线翻转
    if random.choice([1,2]) == 1:
        image = tf.image.transpose_image(image)
    # 色彩
    op_type = random.choice([None, 0, 1, 2, 3, 4, 5])
    image = color_adjust(image, op_type=None)
    # 裁剪与缩放
    #已完成
    image = tf.image.resize_images(image, image_size)
    return image

# 定义输入图像大小
image_size = [227, 227]  # [new_height, new_width]
batch_size = 256
shuffle_buffer = 10000
NUM_EPOCHS = 40

### 从TFRecord文件创建数据集
train_input_files = tf.placeholder(tf.string)
val_input_files = tf.placeholder(tf.string)
test_input_files = tf.placeholder(tf.string)
train_files = tf.train.match_filenames_once(train_input_files)
val_files = tf.train.match_filenames_once(val_input_files)
test_files = tf.train.match_filenames_once(test_input_files)
# 定义数据集
# 训练数据集
train_dataset = tf.data.TFRecordDataset(train_files)
train_dataset = train_dataset.map(parser)
# 验证数据集
val_dataset = tf.data.TFRecordDataset(val_files)
val_dataset = val_dataset.map(parser).map(lambda image, label : (tf.image.resize_images(image, image_size), label))
val_dataset = val_dataset.batch(batch_size)
# 测试数据集
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser).map(lambda image, label : (tf.image.resize_images(image, image_size), label))
test_dataset = test_dataset.batch(batch_size)

# 依次对训练集数据进行 预处理, shuffle, batching；对验证集、测试集无须处理
train_dataset = train_dataset.map(lambda image, label : (preprocess(image, image_size), label))
train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)

# NUM_EPOCHS训练数据集重复的次数
train_dataset = train_dataset.repeat(NUM_EPOCHS)

# 定义数据集的迭代器
# 训练集
train_iterator = train_dataset.make_initializable_iterator()
train_images, train_labels = train_iterator.get_next()
# 验证集
val_iterator = val_dataset.make_initializable_iterator()
val_images, val_labels = val_iterator.get_next()
# 测试集
test_iterator = test_dataset.make_initializable_iterator()
test_images, test_labels = test_iterator.get_next()

#########################################################
########### 定义神经网络模型  ##############
""" end in line: 345
"""
# get current time as ***.cpkt files' name(identifier)
def cur_time():
    ct = time.localtime()
    ct_str = str(ct[0])
    for i in range(1, 6):
        ct_str += '%02d' % ct[i]
    return ct_str

## setup AlexNet

# define layers
def inference(input_tensor, op_type='train'):
    """ op_type: one of {train, val, test}
    """
    # 
    #with tf.name_scope('means_vars'):
    #    mean1 = tf.Variable(initial_value=tf.zeros([1, 27, 27, 96]), trainable=False, name='mean1')
    #    var1 = tf.Variable(initial_value=tf.zeros([1, 27, 27, 96]), trainable=False, name='var1')
    #    mean2 = tf.Variable(initial_value=tf.zeros([1, 13, 13, 256]), trainable=False, name='mean2')
    #    var2 = tf.Variable(initial_value=tf.zeros([1, 13, 13, 256]), trainable=False, name='var2')
    #    tf.summary.histogram('mean1', mean1)
    #    tf.summary.histogram('var1', var1)

    w_initializer = tf.random_normal_initializer(stddev=0.01)
    b_initializer = tf.ones_initializer()
    # layer 1: conv 1
    # filter [11, 11], S = 4
    with tf.variable_scope('layer_1_conv'):
        w1 = tf.get_variable("weights", [11, 11, 3, 96], initializer=w_initializer)
        b1 = tf.get_variable("bias", [96], initializer=tf.zeros_initializer())
        h_conv1 = tf.nn.conv2d(
            input=input_tensor,
            filter=w1,
            strides=[1, 4, 4, 1],
            padding='VALID',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name='conv'
        ) + b1
        # relu
        a1 = tf.nn.relu(h_conv1)
        #
        tf.summary.histogram('weights', w1)
        tf.summary.histogram('biases', b1)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w1)
        
    # layer 1: max pool 1
    with tf.name_scope('layer_1_pool'):
        h_pool1 = tf.nn.max_pool(
            value=a1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            data_format='NHWC',
            name='max_pool'
        )
    
    # layer 1: normalization 1
    #with tf.name_scope('layer_1_norm'):
    #    mean1 = tf.reduce_mean(h_pool1, axis=0, keepdims=True)
    #    var1 = tf.reduce_mean((h_pool1 - mean1)**2, axis=0, keepdims=True)
    #    h_norm1 = tf.nn.batch_normalization(
    #        x=h_pool1,
    #        mean=mean1,
    #        variance=var1,
    #        offset=0,
    #        scale=1,
    #        variance_epsilon=1e-8
    #    )

    # layer 2: conv 2
    with tf.variable_scope('layer_2_conv'):
        w2 = tf.get_variable("weights", [5, 5, 96, 256], initializer=w_initializer)
        b2 = tf.get_variable("bias", [256], initializer=b_initializer)
        h_conv2 = tf.nn.conv2d(
            input=h_pool1,
            filter=w2,
            strides=[1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name='conv'
        ) + b2
        # relu
        a2 = tf.nn.relu(h_conv2)
        #
        tf.summary.histogram('weights', w2)
        tf.summary.histogram('biases', b2)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w2)

    # layer 2: max pool 2
    with tf.name_scope('layer_2_pool'):
        h_pool2 = tf.nn.max_pool(
            value=a2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            data_format='NHWC',
            name='max_pool'
        )

    # layer 2: normalization 2
    #with tf.name_scope('layer_2_norm'):
    #    mean2 = tf.reduce_mean(h_pool2, axis=0, keepdims=True)
    #    var2 = tf.reduce_mean((h_pool2 - mean2)**2, axis=0, keepdims=True)
    #    h_norm2 = tf.nn.batch_normalization(
    #        x=h_pool2,
    #        mean=mean2,
    #        variance=var2,
    #        offset=0,
    #        scale=1,
    #        variance_epsilon=1e-8
    #    )

    # layer 3: conv 3
    with tf.variable_scope('layer_3_conv'):
        w3 = tf.get_variable("weights", [3, 3, 256, 384], initializer=w_initializer)
        b3 = tf.get_variable("bias", [384], initializer=tf.zeros_initializer())
        h_conv3 = tf.nn.conv2d(
            input=h_pool2,
            filter=w3,
            strides=[1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name='conv'
        ) + b3
        # relu
        a3 = tf.nn.relu(h_conv3)
        #
        tf.summary.histogram('weights', w3)
        tf.summary.histogram('biases', b3)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w3)

    # layer 4: conv 4
    with tf.variable_scope('layer_4_conv'):
        w4 = tf.get_variable("weights", [3, 3, 384, 384], initializer=w_initializer)
        b4 = tf.get_variable("bias", [384], initializer=b_initializer)
        h_conv4 = tf.nn.conv2d(
            input=a3,
            filter=w4,
            strides=[1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name='conv'
        ) + b4
        # relu
        a4 = tf.nn.relu(h_conv4)
        #
        tf.summary.histogram('weights', w4)
        tf.summary.histogram('biases', b4)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w4)

    # layer 5: conv 5
    with tf.variable_scope('layer_5_conv'):
        w5 = tf.get_variable("weights", [3, 3, 384, 256], initializer=w_initializer)
        b5 = tf.get_variable("bias", [256], initializer=tf.ones_initializer())
        h_conv5 = tf.nn.conv2d(
            input=a4,
            filter=w5,
            strides=[1, 1, 1, 1],
            padding='SAME',
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            name='conv'
        ) + b5
        # relu
        a5 = tf.nn.relu(h_conv5)
        #
        tf.summary.histogram('weights', w5)
        tf.summary.histogram('biases', b5)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w5)

    # layer 5: max pool 3
    with tf.name_scope('layer_5_pool'):
        h_pool3 = tf.nn.max_pool(
            value=a5,
            ksize=[1, 2, 2,1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            data_format='NHWC',
            name='max_pool'
        )
        # reshape
        a_pool3 = tf.reshape(h_pool3, [-1, 6*6*256])

    # layer 6: FC 6
    with tf.variable_scope('layer_6_FC'):
        w6 = tf.get_variable('weights', [6*6*256, 4096], initializer=w_initializer)
        b6 = tf.get_variable('bias', [4096], initializer=b_initializer)
        h_fc6 = tf.matmul(a_pool3, w6) + b6
        h6 = tf.nn.relu(h_fc6)
        #
        tf.summary.histogram('weights', w6)
        tf.summary.histogram('biases', b6)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w6)

    # layer 7: FC 7
    with tf.variable_scope('layer_7_FC'):
        w7 = tf.get_variable('weights', [4096, 4096], initializer=w_initializer)
        b7 = tf.get_variable('bias', [4096], initializer=b_initializer)
        h_fc7 = tf.matmul(h6, w7) + b7
        h7 = tf.nn.relu(h_fc7)
        #
        tf.summary.histogram('weights', w7)
        tf.summary.histogram('biases', b7)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w7)

    # layer 8: FC 8
    with tf.variable_scope('layer_8_FC'):
        w8 = tf.get_variable('weights', [4096, 1000], initializer=w_initializer)
        b8 = tf.get_variable('bias', [1000], initializer=b_initializer)
        h_fc8 = tf.matmul(h7, w8) + b8
        #
        tf.summary.histogram('weights', w8)
        tf.summary.histogram('biases', b8)
        #
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w8)
    
    return h_fc8

# input data placeholder
with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='image_batch')
    y = tf.placeholder(tf.int64, shape=[None], name='labels')

# get inference: logits
logits = inference(X)
#val_logits = tf.argmax(logits, -1)
# loss
with tf.name_scope('loss') as scope:
    # regularization loss
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    regularization_loss = tf.contrib.layers.apply_regularization(regularizer=regularizer)
    # softmax loss
    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y, 1000), logits=logits)
    # total loss
    loss = regularization_loss + cross_entropy_loss
    #
    tf.summary.scalar('loss', loss)

#
with tf.name_scope('accuarcy'):
    correct_prediction = tf.equal(tf.argmax(logits, -1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# global step
global_step = tf.Variable(0, trainable=False, name='global_step')

# train op
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.9
with tf.name_scope('optimizer'):
    learning_rate = tf.train.exponential_decay(
                        LEARNING_RATE_BASE,
                        global_step,
                        int(1261174 / batch_size + 1) ,  #4927. total train images are: 1261174
                        LEARNING_RATE_DECAY,
                        staircase = True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

# merge all summaries
merge_all = tf.summary.merge_all()

# to save model
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=10000.0)
MODEL_PATH = 'path/to/model/'

# training steps
train_filename = '/path/to/train-tfrecords/train-*'
val_filename = '/path/to/val-tfrecords/val.tfrecord'
test_filename = '/path/to/test-tfrecords/test.tfrecord'
#num_epoch = NUM_EPOCHS
#num_test = 50000
#num_val = 10000
#num_iteration = 1280000 // 128
RESTORE = False
# tf.session config
config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    # restore from model.skpt or initialize from the scratch
    if RESTORE:
        saver.restore(sess, MODEL_PATH + 'model.ckpt')
    else:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()], feed_dict={
                train_input_files: train_filename,
                val_input_files: val_filename,
                test_input_files: test_filename})
        sess.run([train_iterator.initializer, val_iterator.initializer, test_iterator.initializer])
    
        train_writer = tf.summary.FileWriter('/path/to/log/alexnet/train/', sess.graph)
        val_writer = tf.summary.FileWriter('/path/to/log/alexnet/val/', sess.graph)
        # train process
        step = 0
        val_step = 0
    ###
    while True:
        try:
            #print("Epoch %d" % (step//2464))
            #t1 = time.time()
            image_batch, label_batch = sess.run([train_images, train_labels])
            if step % 200 == 0:
                # val and summary
                val_step = 1
                val_acc = 0
                while True:
                    try:
                        val_image_batch, val_label_batch = sess.run([val_images, val_labels])
                        summary, val_correct = sess.run([merge_all, accuracy], feed_dict={X: val_image_batch, y: val_label_batch})
                        val_writer.add_summary(summary, step)
                        val_acc += val_correct
                        val_step += 1
                    except tf.errors.OutOfRangeError:
                        break
                val_acc /= val_step
                print('%d iteration(s), validation accuracy is: %.5f' % (step, val_acc))
                # save model
                if step != 0:
                    MODEL_NAME = 'model-%s.ckpt' % cur_time()
                    print('step is %d' % step)
                    saver.save(sess, MODEL_PATH + MODEL_NAME, global_step=step)
                # train and summary
                # 配置运行时需要记录的信息和记录运行信息的proto
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, train_loss, _  = sess.run([merge_all, loss, train_op], feed_dict={X: image_batch, y: label_batch}, options=run_options, run_metadata=run_metadata)
                train_writer.add_summary(summary, step)
                train_writer.add_run_metadata(run_metadata, 'step%d' % step)
            else:
                # train
                summary, train_loss, _ = sess.run([merge_all, loss, train_op], feed_dict={X: image_batch, y: label_batch})
                train_writer.add_summary(summary, step)
            print('iteration: %d,  loss: %f' % (step, train_loss))
            step += 1
        except tf.errors.OutOfRangeError:
            print('train process end at %f ' % (time.time()-t1))
            break
    train_writer.close()
    val_writer.close()

    # test process
    test_logits = []
    while True:
        try:
            # test accuracy
            test_image_batch, test_label_batch = sess.run([test_images, test_labels])
            test_log = sess.run([logits], feed_dict={X: test_image_batch, y: test_label_batch})
            test_logits.extend(test_log)
        except tf.errors.OutOfRangeError:
            test_logits = np.array(test_logits)
            test_logits.tofile('./prediction_logits.npy')
            print("predictions are in \"./prediction_logits.npy\".")
            print('test end at: %f...' % (time.time()-t1))
            break





