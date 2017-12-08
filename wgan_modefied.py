import os
import time
import argparse
import importlib
from scipy.misc import imsave


import tensorflow as tf
from utils import load_rand_data_label,load_rand_data,load_rand_label
import numpy as np
import random
import gc
import time
from PIL import Image

def load_distinguish_data(my_generate_data = None):
    # my_generate_data = np.asarray(my_generate_data,np.float32)
    SELECT_NUM = 10000
    my_data_1 = load_rand_label(SELECT_NUM)
    my_data_2 = load_rand_data(SELECT_NUM,my_load_level = 1)
    my_data_3 = load_rand_data(SELECT_NUM,my_load_level = 2)
    my_data = np.concatenate([my_data_1, my_data_2, my_data_3, my_generate_data], axis=0)
    del my_data_1,my_data_2,my_data_3
    gc.collect()

    my_label = np.zeros([SELECT_NUM*4,4], dtype='float32')
    my_label[SELECT_NUM*0:SELECT_NUM*1 ,0] = 1.0
    my_label[SELECT_NUM*1:SELECT_NUM*2 ,1] = 1.0
    my_label[SELECT_NUM*2:SELECT_NUM*3 ,2] = 1.0
    my_label[SELECT_NUM*3:SELECT_NUM*4 ,3] = 1.0

    print my_label.shape
    return my_data,my_label

def load_SR_data(SELECT_NUM = 10000):
    my_data, my_label = load_rand_data_label(SELECT_NUM,my_load_level=2)
    print  my_data.shape
    return my_data,my_label

class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

        self.chananel_num = [128,64,32]
        self.output_channel_num = [256,64,3]

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs

            with tf.variable_scope('conv1'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                inputs = x_internal
                x_res = inputs
                x_intial_res = inputs

            with tf.variable_scope('conv2'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv3'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv4'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv5'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                inputs = x_internal

            with tf.variable_scope('Re'):
                x_internal = inputs + x_intial_res
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(x_internal, name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(x_internal, name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.sigmoid(x_internal, name='l_sigmoid')
                inputs = x_internal

        outputs = inputs
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths
        self.reuse = False
        self.chananel_num = [16, 16, 16, 1, 1]

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        inputs = tf.convert_to_tensor(inputs)


        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4

            with tf.variable_scope('conv1'):
                c1 = tf.layers.conv2d(inputs, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c3_outputs1')
                input = c1


            with tf.variable_scope('conv2'):
                c2 = tf.layers.conv2d(input, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                c2 = leaky_relu(tf.layers.batch_normalization(c2, training=training), name='c3_outputs1')
                input = c2


            with tf.variable_scope('conv3'):
                c3 = tf.layers.conv2d(input, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                input = c3

            with tf.variable_scope('conv4'):
                c4 = tf.layers.conv2d(input, self.chananel_num[3], [3, 3], strides=(1, 1), padding='SAME')
                c4 = leaky_relu(tf.layers.batch_normalization(c4, training=training), name='c3_outputs1')


            with tf.variable_scope('classify'):
                # outputs = tf.concat([c1, c4],axis=3)
                outputs = c4

                # outputs = tf.layers.conv2d(input, self.chananel_num[4], [1, 1], strides=(1, 1), padding='SAME')
                # outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs1')

                # outputs = c3
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, input_tensor, label_tensor,batch_size = 10, g_lr = 1e-4,d_lr = 1e-4 ,scale=10.0):
        self.save_path = '../MODEL/WGAN/'
        self.batch_size = batch_size
        self.g_net = g_net
        self.d_net = d_net
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.x = label_tensor
        self.z = input_tensor

        self.test_img_input = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                             name='input_tensor')
        self.test_img_out = self.g_net(self.test_img_input, training=True)

        self.x_ = self.g_net(self.z, training=True)

        self.d = self.d_net(self.x, training=True,name = 'dd')
        self.d_ = self.d_net(self.x_,training=True, name= 'dg')

        self.g_loss = tf.reduce_mean(tf.square(self.d - self.d_))
        self.d_loss = tf.reduce_mean(-tf.square(self.d - self.d_))

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        # ddx = tf.gradients(d_hat, x_hat)[0]
        # print(ddx.get_shape().as_list())
        # ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        # ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        #
        # self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None

        d_global_step = tf.Variable(0, trainable=False)
        d_lr_rate = tf.train.exponential_decay(self.d_lr, d_global_step, 5000, 0.9, staircase=True)

        g_global_step = tf.Variable(0, trainable=False)
        g_lr_rate = tf.train.exponential_decay(self.g_lr, g_global_step, 10000, 0.9, staircase=True)

        # self.d_adam = tf.train.AdamOptimizer(learning_rate=d_lr_rate, beta1=0.4) \
        #     .minimize(self.d_loss, var_list=self.d_net.variables, global_step=d_global_step)
        # self.g_adam = tf.train.AdamOptimizer(learning_rate=g_lr_rate, beta1=0.4) \
        #     .minimize(self.g_loss, var_list=self.g_net.variables, global_step=g_global_step)

        self.d_adam = tf.train.AdamOptimizer(learning_rate=d_lr_rate, beta1=0.4) \
            .minimize(self.d_loss, var_list=self.d_net.variables, global_step=d_global_step)
        self.g_adam = tf.train.AdamOptimizer(learning_rate=g_lr_rate, beta1=0.8) \
            .minimize(self.g_loss, var_list=self.g_net.variables, global_step=g_global_step)


        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

        # self.adam_opt = tf.control_dependencies([self.d_adam,self.g_adam])

        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
        #         .minimize(self.d_loss, var_list=self.d_net.variables)
        #     self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
        #         .minimize(self.g_loss, var_list=self.g_net.variables)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, input_data,input_label,batch_size=10, iterations=1000000, RESTORED = False):
        a1, a2, a3, a4 = input_data.shape
        DATA_NUM = a1

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if RESTORED:
            self.saver.restore(self.sess,self.save_path)

        start_time = time.time()
        for t in range(0, iterations):
            t=t+1
            d_iters = 2
            # for _ in range(0, d_iters):
            #     SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            #     batch_input = input_data[SELECT_LIST, :, :, :]
            #     # SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            #     batch_labels = input_label[SELECT_LIST, :, :, :]
            #     self.sess.run(self.d_adam, feed_dict={self.x: batch_labels, self.z: batch_input})

            SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            batch_input = input_data[SELECT_LIST, :, :, :]
            # SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            batch_labels = input_label[SELECT_LIST, :, :, :]
            self.sess.run([self.d_adam, self.g_adam], feed_dict={self.z: batch_input, self.x: batch_labels})

            if t % 200 == 0:
                SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
                batch_input = input_data[SELECT_LIST, :, :, :]
                batch_labels = input_label[SELECT_LIST, :, :, :]

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: batch_labels, self.z: batch_input}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: batch_input, self.x: batch_labels}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%4.4f] g_loss [%4.4f]' %
                        (t, time.time() - start_time, d_loss, g_loss))

            if t % 1000 == 0:
                print  'saving test imgs '
                self.saver.save(self.sess,self.save_path)
                self.test_img(self.sess,save_name='wgan%d'%(t),my_level=2)

    def test_img(self,sess,save_name,my_level = 2):
        level = my_level
        imag_path = '../../../SOURCE/Img/Set14/comic.bmp'
        imag_save_path = '../RESULT/WGAN/X' + str(my_level) + '/'

        test_Image = Image.open(imag_path)
        test_Image.save(imag_save_path + 'Original.bmp')

        w, h = test_Image.size
        w_n = (w / level)
        h_n = (h / level)
        my_size = w, h
        my_low_size = w_n, h_n

        test_Image = test_Image.resize(my_low_size, Image.BICUBIC)
        test_Image = test_Image.resize(my_size, Image.BICUBIC)
        test_Image.save(imag_save_path + 'input.bmp')

        pixels = np.array(test_Image, dtype=np.double)
        pixels.astype(np.double)
        pixels = pixels / 255.0

        input_image = np.array(pixels, dtype='double')
        input_image = input_image[np.newaxis, :, :, :]
        # my_input = tf.placeholder(tf.float32, [None, h, w, 3], name='images')

        output_img = sess.run([self.test_img_out], feed_dict={self.test_img_input: input_image})

        output_img = output_img[0]
        result_image = np.array(output_img, dtype='double')
        print result_image.shape
        result_image = result_image[0, :, :, :]
        # result_image = result_image * (2.0) - np.ones(result_image.shape,dtype=np.double)
        print result_image.shape
        # result_image = result_image + pixels
        result_image = result_image * 255.0
        result_image = np.ceil(result_image)
        result_image = result_image.astype('uint8')
        print(result_image.dtype)
        img = Image.fromarray(result_image, mode='RGB')

        img.save(imag_save_path + save_name + 'output.bmp')
        return


if __name__ == '__main__': 

    batch_size = 20
    SELECT_NUM = 14000
    MAX_NUM_ITERARION = 500000
    d_lr = 1e-5
    g_lr = 1e-5

    d_net = Discriminator()
    g_net = Generator()

    img = tf.placeholder(tf.float32, shape=[batch_size, 80, 80, 3],
                         name='input_tensor')
    label = tf.placeholder(tf.float32, shape=[batch_size, 80, 80, 3],
                           name='output_tensor')

    wgan = WassersteinGAN(g_net, d_net,img,label,batch_size=batch_size,g_lr=g_lr,d_lr=d_lr)

    my_data, my_label = load_SR_data(SELECT_NUM=SELECT_NUM)

    wgan.train(input_data=my_data,input_label=my_label,batch_size=batch_size,iterations = MAX_NUM_ITERARION)
