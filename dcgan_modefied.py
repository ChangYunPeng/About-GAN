import tensorflow as tf
from utils import load_rand_data_label,load_rand_data,load_rand_label
import numpy as np
import random
import gc
import time
from PIL import Image
from Generator import Slim_Generator as Generator
from Discriminator import FullSize_C1C3_Discriminator as Discriminator

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

class DCGAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 g_depths=[1024, 512, 256, 128],
                 d_depths=[64, 128, 256, 512]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = Generator(depths=g_depths, s_size=self.s_size)
        self.d = Discriminator(depths=d_depths)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

        self.test_img_input = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                  name='input_tensor')
        self.test_img_out = self.g(self.test_img_input,training = True)

    def pre_train_loss(self, img,label):
        generated = self.g(img, training=True)
        self.mse_loss = tf.reduce_mean(tf.square(generated - label))
        g_mse_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5)
        self.g_mse_opt_op = g_mse_opt.minimize(self.mse_loss, var_list=self.g.variables)
        return

    def loss(self, img,label):
        """build models, calculate losses.

        Args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.

        Returns:
            dict of each models' losses.
        """
        generated = self.g(img, training=True)
        g_outputs = self.d(generated, training=True, name='g')
        t_outputs = self.d(label, training=True, name='t')
        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))



        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)) )

        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)) )
        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def train(self, losses, g_learning_rate=1e-5,d_learning_rate=1e-5, beta1=0.5):
        """
        Args:
            losses dict.

        Returns:
            train op.
        """
        g_opt = tf.train.AdamOptimizer(learning_rate=g_learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=d_learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)

        tf.summary.scalar('D_Loss',losses[self.d])
        tf.summary.scalar('G_Loss',losses[self.g])

        self.d_opt_apply = d_opt_op
        self.g_opt_apply = g_opt_op
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = self.g(inputs, training=True)
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))

    def test_img(self,sess,save_name,my_level = 2):
        level = my_level
        imag_path = '../../../SOURCE/Img/Set14/comic.bmp'
        imag_save_path = '../RESULT/X' + str(my_level) + '/'

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

    batch_size = 8
    SELECT_NUM = 14000
    pre_train_steps = 500
    d_steps = 1
    g_steps = 1

    dcgan = DCGAN(batch_size=batch_size)
    img = tf.placeholder(tf.float32, shape=[batch_size, 80, 80, 3],
                         name='input_tensor')
    label = tf.placeholder(tf.float32,shape=[batch_size, 80, 80, 3],
                                            name='output_tensor')


    my_data, my_label = load_SR_data(SELECT_NUM = SELECT_NUM)



    ''' Pre - Train '''
    # dcgan.pre_train_loss(img, label)
    # start_time = time.time()
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     for step in range(max_steps):
    #         SELECT_LIST = np.array(random.sample(range(0, SELECT_NUM), batch_size), dtype='int')
    #         batch_input = my_data[SELECT_LIST, :, :, :]
    #         batch_labels = my_label[SELECT_LIST, :, :, :]
    #
    #         _, losses = sess.run([dcgan.g_mse_opt_op, dcgan.mse_loss], feed_dict={img:batch_input,label:batch_labels})
    #
    #
    #         if (step + 1)%500 == 0:
    #             print ('Steps%d,    Losses:%.8f,    time-used:%8.2f') %(step+1,losses,time.time() - start_time)
    #             print saver.save(sess,save_path)

    dc_loss = dcgan.loss(img,label)


    opt = dcgan.train(dc_loss)

    saver = tf.train.Saver()
    save_path = '../MODEL/DCGAN/'

    summaries_dir = '../MODEL/DCGAN/SINGLE_GPU%d.CPTK'%time.time()


    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir,
                                         graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, save_path)
        start_time = time.time()
        for step in range(pre_train_steps):
            SELECT_LIST = np.array(random.sample(range(0, SELECT_NUM), batch_size), dtype='int')
            batch_input = my_data[SELECT_LIST, :, :, :]
            batch_labels = my_label[SELECT_LIST, :, :, :]

            _, losses_d = sess.run([dcgan.d_opt_apply,  dc_loss[dcgan.d]],
                                 feed_dict={img: batch_input, label: batch_labels})

            if (step + 1) % 200 == 0:
                print ('pre-train DCGAN -- Steps%d,    time-used:%8.2f') % (step + 1,  time.time() - start_time)
                print ('          D Loss:%4.4f') % losses_d
            if (step + 1) % 500 == 0:
                print saver.save(sess, save_path)
                # dcgan.test_img(sess,save_name='dcgan%d'%(step+1),my_level=2)
        ''' '''
        print 'Pre Train Finished'
        print saver.save(sess, save_path)

        start_time = time.time()
        for step in range(max_steps):

            for d_step in range(d_steps):
                SELECT_LIST = np.array(random.sample(range(0, SELECT_NUM), batch_size), dtype='int')
                batch_input = my_data[SELECT_LIST, :, :, :]
                batch_labels = my_label[SELECT_LIST, :, :, :]

                _,  losses_d = sess.run([dcgan.d_opt_apply, dc_loss[dcgan.d]],
                                                 feed_dict={img: batch_input, label: batch_labels})

            for g_step in range(g_steps):
                SELECT_LIST = np.array(random.sample(range(0, SELECT_NUM), batch_size), dtype='int')
                batch_input = my_data[SELECT_LIST, :, :, :]
                batch_labels = my_label[SELECT_LIST, :, :, :]

                _,  losses_g = sess.run([dcgan.g_opt_apply, dc_loss[dcgan.g] ],
                                                 feed_dict={img: batch_input, label: batch_labels})

            if (step + 1) % 200 == 0:
                print ('DCGAN -- Steps%d,    time-used:%8.2f') % (step + 1,  time.time() - start_time)
                print ('D Loss:%4.4f') % losses_d
                print ('G Loss:%4.4f') % losses_g


            if (step + 1) % 1000 == 0:
                SELECT_LIST = np.array(random.sample(range(0, SELECT_NUM), batch_size), dtype='int')
                batch_input = my_data[SELECT_LIST, :, :, :]
                batch_labels = my_label[SELECT_LIST, :, :, :]

                sum_tmp, _, _ = sess.run([merged, dc_loss[dcgan.d],dc_loss[dcgan.d]],
                                       feed_dict={img: batch_input, label: batch_labels})

                train_writer.add_summary(sum_tmp, step)
                train_writer.flush()

                print saver.save(sess, save_path)
                dcgan.test_img(sess,save_name='dcgan%d'%(step+1),my_level=2)

    return


run(400000)