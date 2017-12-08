import tensorflow as tf

class C1C3_Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths
        self.reuse = False
        self.chananel_num = [4, 16, 64, 128]

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        c3 = outputs
        c1 = outputs

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4

            with tf.variable_scope('conv1'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                c3 = tf.layers.conv2d(c3, self.chananel_num[0], [2, 2], strides=(2, 2), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                c1 = tf.layers.conv2d(c1, self.chananel_num[0], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                c1 = tf.layers.conv2d(c1, self.chananel_num[0], [2, 2], strides=(2, 2), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv2'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                c3 = tf.layers.conv2d(c3, self.chananel_num[1], [2, 2], strides=(2, 2), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                c1 = tf.layers.conv2d(c1, self.chananel_num[1], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                c1 = tf.layers.conv2d(c1, self.chananel_num[1], [2, 2], strides=(2, 2), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv3'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                c3 = tf.layers.conv2d(c3, self.chananel_num[2], [2, 2], strides=(2, 2), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                c1 = tf.layers.conv2d(c1, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                c1 = tf.layers.conv2d(c1, self.chananel_num[2], [2, 2], strides=(2, 2), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv4'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[3], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                c3 = tf.layers.conv2d(c3, self.chananel_num[3], [2, 2], strides=(2, 2), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                c1 = tf.layers.conv2d(c1, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                c1 = tf.layers.conv2d(c1, self.chananel_num[3], [2, 2], strides=(2, 2), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('classify'):
                batch_size = c3.get_shape()[0].value
                reshape_c3 = tf.reshape(c3, [batch_size, -1])
                reshape_c1 = tf.reshape(c1, [batch_size, -1])
                reshape = tf.concat([reshape_c3,reshape_c1],axis=1)
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

class FullSize_C1C3_Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths
        self.reuse = False
        self.chananel_num = [64, 16, 4, 1]

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        c3 = outputs
        c1 = outputs

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4

            with tf.variable_scope('conv1'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')

            with tf.variable_scope('conv2'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')

            with tf.variable_scope('conv3'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')

            with tf.variable_scope('conv4'):
                c3 = tf.layers.conv2d(c3, self.chananel_num[3], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')

                c1 = tf.layers.conv2d(c1, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')

            with tf.variable_scope('classify'):
                batch_size = c3.get_shape()[0].value
                reshape_c3 = tf.reshape(c3, [batch_size, -1])
                reshape_c1 = tf.reshape(c1, [batch_size, -1])
                reshape = tf.concat([reshape_c3,reshape_c1],axis=1)
                outputs = tf.layers.dense(reshape, 2, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

class C3_Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.depths = [3] + depths
        self.reuse = False
        self.chananel_num_c3 = [64, 16, 4, 1, 1]
        self.chananel_num_c1 = [8, 4, 2, 1, 1]

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        inputs = tf.convert_to_tensor(inputs)


        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            c3 = inputs
            c1 = inputs

            with tf.variable_scope('conv1'):
                c3 = tf.layers.conv2d(c3, self.chananel_num_c3[0], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num_c3[0], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[0], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num_c1[0], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv2'):
                c3 = tf.layers.conv2d(c3, self.chananel_num_c3[1], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num_c3[1], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[1], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num_c1[1], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv3'):
                c3 = tf.layers.conv2d(c3, self.chananel_num_c3[2], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num_c3[2], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                # c1 = tf.layers.conv2d(c1, self.chananel_num[2], [1, 1], strides=(1, 1), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num_c1[2], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('conv4'):
                c3 = tf.layers.conv2d(c3, self.chananel_num_c3[3], [3, 3], strides=(1, 1), padding='SAME')
                c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs1')
                # c3 = tf.layers.conv2d(c3, self.chananel_num_c3[3], [2, 2], strides=(2, 2), padding='SAME')
                # c3 = leaky_relu(tf.layers.batch_normalization(c3, training=training), name='c3_outputs2')

                c1 = tf.layers.conv2d(c1, self.chananel_num_c1[3], [1, 1], strides=(1, 1), padding='SAME')
                c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs1')
                # c1 = tf.layers.conv2d(c1, self.chananel_num_c1[3], [2, 2], strides=(2, 2), padding='SAME')
                # c1 = leaky_relu(tf.layers.batch_normalization(c1, training=training), name='c1_outputs2')

            with tf.variable_scope('classify'):
                # outputs = tf.concat([c1, c4],axis=3)
                # outputs = tf.concat([c1, c3],axis=3)
                # outputs = tf.layers.dense(outputs, 2, name='outputs')

                batch_size = c3.get_shape()[0].value
                reshape_c3 = tf.reshape(c3, [batch_size, -1])
                reshape_c1 = tf.reshape(c1, [batch_size, -1])
                reshape = tf.concat([reshape_c3, reshape_c1], axis=1)
                outputs = tf.layers.dense(reshape, 2, name='outputs')

                # outputs = tf.layers.conv2d(input, self.chananel_num[4], [1, 1], strides=(1, 1), padding='SAME')
                # outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs1')

                # outputs = c3
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs
