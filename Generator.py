import tensorflow as tf

class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

        self.chananel_num = [128,128,64,32]
        self.output_channel_num = [256,64,3]

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs

            with tf.variable_scope('conv1'):
                former_inputs = inputs
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l4')
                x_outputs = x_internal
                x_intial_res = x_internal

            with tf.variable_scope('conv2'):
                x_internal = x_outputs
                former_inputs = x_internal
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l4')
                x_outputs = x_internal

            with tf.variable_scope('conv3'):
                x_internal = x_outputs + former_inputs
                former_inputs = x_internal
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l4')
                x_outputs = x_internal

            with tf.variable_scope('conv4'):
                x_internal = x_outputs + former_inputs
                former_inputs = x_internal
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l4')
                x_outputs = x_internal

            with tf.variable_scope('conv5'):
                x_internal = x_outputs + former_inputs
                former_inputs = x_internal
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [3, 3], strides=(1, 1), padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[3], [1, 1], strides=(1, 1), padding='SAME')
                # x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l4')
                x_outputs = x_internal

            with tf.variable_scope('Re'):
                x_internal = x_outputs + x_intial_res
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

class Slim_Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.reuse = False

        self.chananel_num = [128, 64, 32]
        self.output_channel_num = [256, 64, 3]

    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs

            with tf.variable_scope('conv1'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                inputs = x_internal
                x_res = inputs
                x_intial_res = inputs

            with tf.variable_scope('conv2'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv3'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv4'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                x_res = x_internal + x_res
                inputs = x_res

            with tf.variable_scope('conv5'):
                x_internal = inputs
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.chananel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(tf.layers.batch_normalization(x_internal, training=training), name='l3')
                inputs = x_internal

            with tf.variable_scope('Re'):
                x_internal = inputs + x_intial_res
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[0], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(x_internal, name='l1')
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[1], [3, 3], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.relu(x_internal, name='l2')
                x_internal = tf.layers.conv2d(x_internal, self.output_channel_num[2], [1, 1], strides=(1, 1),
                                              padding='SAME')
                x_internal = tf.nn.sigmoid(x_internal, name='l_sigmoid')
                inputs = x_internal

        outputs = inputs
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs