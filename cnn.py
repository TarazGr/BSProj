import tensorflow as tf

# MODEL #

def siamese_network(img, reuse_variables=False):

    with tf.name_scope('siamese'):

        with tf.variable_scope('conv1') as scope:
            layer = tf.contrib.layers.conv2d(inputs=img, num_outputs=32, kernel_size=[10, 10], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv2') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=64, kernel_size=[7, 7], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv3') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=64, kernel_size=[4, 4], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('conv4') as scope:
            layer = tf.contrib.layers.conv2d(inputs=layer, num_outputs=128, kernel_size=[4, 4], padding='VALID', activation_fn=tf.nn.relu,
                                             biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01), scope=scope,
                                             reuse=reuse_variables)
            layer = tf.contrib.layers.max_pool2d(layer, kernel_size=[2, 2], stride=2, padding='VALID')

        with tf.variable_scope('flatten') as scope:
            layer = tf.contrib.layers.flatten(inputs=layer)

        with tf.variable_scope('fc') as scope:
            layer = tf.contrib.layers.fully_connected(inputs=layer, num_outputs=4096, activation_fn=tf.nn.sigmoid,
                                                      biases_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.01),
                                                      scope=scope, reuse=reuse_variables)

        return layer
