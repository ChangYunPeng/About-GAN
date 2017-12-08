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
    def __init__(self):

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

class Color_Restore:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

        self.chananel_num = [128,64,32]
        self.output_channel_num = [256,64,3]

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)

        with tf.variable_scope('resotre', reuse=self.reuse):
            # reshape from inputs
            x_internal = inputs
            x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[0], [1, 1], strides=(1, 1), padding='SAME')
            x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
            x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[1], [1, 1], strides=(1, 1), padding='SAME')
            x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
            x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[2], [1, 1], strides=(1, 1), padding='SAME')
            x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
            inputs = x_internal

        outputs = inputs
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  # print tower_grads
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      if(g!=None):
        expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


class Discriminator:
    def __init__(self ):

        self.reuse = False
        self.chananel_num = [256, 64, 8, 1]

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        c3 = outputs
        # c1 = outputs

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4

            with tf.variable_scope('conv1'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num[0], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[0], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num[0], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv2'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num[1], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[1], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num[1], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv3'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num[2], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num[2], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv4'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[3], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num[3], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num[3], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('classify'):
                batch_size = c3.get_shape()[0].value
                reshape_c3 = tf.reshape(c3, [batch_size, -1])
                # reshape_c1 = tf.reshape(c1, [batch_size, -1])
                # reshape = tf.concat([reshape_c3,reshape_c1],axis=1)
                reshape = reshape_c3
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

class DCGAN:
    def __init__(self,
                 batch_size=20,img_size = 80,g_lr = 1e-4,d_lr = 1e-4, gpu_num = 2):

        self.batch_size = batch_size
        self.img_size = img_size
        self.gpu_num = gpu_num

        self.g_lr = g_lr
        self.d_lr = d_lr

        self.g = Generator()
        self.d = Discriminator()

        self.test_img_input = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='input_tensor')
        self.test_img_out = self.g(self.test_img_input,training = True)

    def set_opt(self):

        g_loss = []
        d_loss = []

        g_opt = []
        d_opt = []

        g_gradients = []
        d_gradients = []

        g_opt_apy = []
        d_opt_apy = []

        with tf.device('/cpu:0'):
            self.img_tensor = tf.placeholder(tf.float32, shape=[self.batch_size*self.gpu_num, self.img_size, self.img_size, 3],
                                        name='input_tensor')
            self.label_tensor = tf.placeholder(tf.float32, shape=[self.batch_size*self.gpu_num, self.img_size, self.img_size, 3],
                                        name='input_tensor')
            img_tensor_slice = tf.split(self.img_tensor, num_or_size_splits=self.gpu_num, axis=0)
            label_tensor_slice = tf.split(self.label_tensor, num_or_size_splits=self.gpu_num, axis=0)



            for i in range(self.gpu_num):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('GPU%d' % i) as scope:
                        x_ = self.g(img_tensor_slice[i])



                        var_num = 0
                        for var in self.g.variables:
                            var_num += 1
                            print 'GPU%d'%i
                            print var
                        print var_num

                        d_ = self.d(x_)
                        d = self.d(label_tensor_slice[i])

                        g_loss.append( tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.ones([self.batch_size], dtype=tf.int64), logits=d_)))

                        d_loss.append( tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.zeros([self.batch_size], dtype=tf.int64), logits=d_)) + tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=tf.ones([self.batch_size], dtype=tf.int64), logits=d)))

                        g_opt.append(tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=0.5))
                        d_opt.append(tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=0.5))

                        g_gradients.append(g_opt[i].compute_gradients(g_loss[i], var_list=self.g.variables))
                        d_gradients.append(d_opt[i].compute_gradients(d_loss[i], var_list=self.d.variables))

                        # g_gradients=(g_opt.compute_gradients(g_loss[i], var_list=self.g.variables))
                        # d_gradients=(d_opt.compute_gradients(d_loss[i], var_list=self.d.variables))

                        # g_opt_apy = g_opt.apply_gradients(g_gradients)
                        # d_opt_apy = d_opt.apply_gradients(d_gradients)

            G_grads = average_gradients(g_gradients)
            D_grads = average_gradients(d_gradients)

            for i in range(self.gpu_num):
                tf.summary.scalar('G_Loss%d'%i,g_loss[i])
                tf.summary.scalar('D_Loss%d'%i,d_loss[i])
            #
            with tf.device('/gpu:0'):
                g_opt_apy = g_opt[0].apply_gradients(G_grads)
                d_opt_apy = d_opt[0].apply_gradients(D_grads)

            # for i in range(self.gpu_num):
            #     with tf.device('/gpu:%d' % i):
            #         with tf.name_scope('GPU%d' % i) as scope:
            #             g_opt_apy.append(g_opt[i].apply_gradients(G_grads))
            #             d_opt_apy.append(d_opt[i].apply_gradients(D_grads))

        print g_opt_apy
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt_apy = g_opt_apy
        self.d_opt_apy = d_opt_apy
        self.img_slice = img_tensor_slice
        self.label_slice = label_tensor_slice
        return

    def train(self,input_data,input_label,d_iters = 5,iterations =10000 ):

        gpu_options = tf.GPUOptions(allow_growth=True)
        summaries_dir = '../MODEL/DCGAN/MULTI_GPU.CPTK'
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        a1, a2, a3, a4 = input_data.shape
        DATA_NUM = a1

        batch_size = self.batch_size * self.gpu_num

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(summaries_dir,
                                             graph=tf.get_default_graph())

        merged = tf.summary.merge_all()

        start_time = time.time()
        for t in range(0, iterations):
            t+=1
            for _ in range(0, d_iters):
                SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
                batch_input = input_data[SELECT_LIST, :, :, :]
                SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
                batch_labels = input_label[SELECT_LIST, :, :, :]
                self.sess.run(self.d_opt_apy, feed_dict={self.img_tensor: batch_input, self.label_tensor: batch_labels})

            SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            batch_input = input_data[SELECT_LIST, :, :, :]
            SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')
            batch_labels = input_label[SELECT_LIST, :, :, :]
            self.sess.run(self.g_opt_apy, feed_dict={self.img_tensor: batch_input, self.label_tensor: batch_labels})

            if t % 200 == 0 and t %1000 != 0:
                SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')

                """ Different """
                batch_input = input_data[SELECT_LIST, :, :, :]
                batch_labels = input_label[SELECT_LIST, :, :, :]


                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.img_tensor: batch_input, self.label_tensor: batch_labels}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.img_tensor: batch_input, self.label_tensor: batch_labels}
                )

                str_title = ('Iter [%8d] Time [%5.4f] ' % (t, time.time() - start_time))
                str_d_loss = str()
                for i_gpu_num in range(self.gpu_num):
                    str_d_loss = str_d_loss + str("D-GPU%d-Loss:%4.8f " % (i_gpu_num, d_loss[i_gpu_num]))
                str_g_loss = str()
                for i_gpu_num in range(self.gpu_num):
                    str_g_loss = str_g_loss + str("G-GPU%d-Loss:%4.8f " % (i_gpu_num, g_loss[i_gpu_num]))
                print str_title + str_d_loss + str_g_loss


            if t % 1000 == 0:
                SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), batch_size), dtype='int')

                """ Different """
                batch_input = input_data[SELECT_LIST, :, :, :]
                batch_labels = input_label[SELECT_LIST, :, :, :]

                sum_tmp, d_loss, g_loss = self.sess.run(
                    [merged,self.d_loss,self.g_loss], feed_dict={self.img_tensor: batch_input, self.label_tensor: batch_labels}
                )

                str_title = ('Iter [%8d] Time [%5.4f] ' % (t, time.time() - start_time))
                str_d_loss = str()
                for i_gpu_num in range(self.gpu_num):
                    str_d_loss = str_d_loss + str("D-GPU%d-Loss:%4.8f " % (i_gpu_num, d_loss[i_gpu_num]))
                str_g_loss = str()
                for i_gpu_num in range(self.gpu_num):
                    str_g_loss = str_g_loss + str("G-GPU%d-Loss:%4.8f " % (i_gpu_num, g_loss[i_gpu_num]))
                print str_title + str_d_loss + str_g_loss

                train_writer.add_summary(sum_tmp, t)
                train_writer.flush()

                print  'saving test imgs '
                self.test_img(self.sess, save_name='dcgan%d' % (t), my_level=2)


    def test_img(self,sess,save_name,my_level = 2):
        level = my_level
        imag_path = '../../../SOURCE/Img/Set14/comic.bmp'
        imag_save_path = '../RESULT/DCGAN/X' + str(my_level) + '/'

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

def run(max_steps = 10000):

    batch_size = 20
    SELECT_NUM = 14000
    pre_train_steps = 500
    d_steps = 2
    g_steps = 1
    dcgan = DCGAN(batch_size=batch_size,gpu_num=2,g_lr = 1e-4,d_lr = 1e-4)
    dcgan.set_opt()


    my_data, my_label = load_SR_data(SELECT_NUM = SELECT_NUM)

    dcgan.train(my_data,my_label,d_steps,max_steps)


run(100000)