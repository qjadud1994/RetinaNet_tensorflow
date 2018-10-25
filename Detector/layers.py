import tensorflow as tf
from tensorflow.contrib import learn

def se_block(bottom, name, ratio=16):
    weight_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    # Bottom [N,H,W,C]
    # Global average pooling
    with tf.variable_scope(name):
        channel = bottom.get_shape()[-1]
        se = tf.reduce_mean(bottom, axis=[1,2], keep_dims=True)
        assert se.get_shape()[1:] == (1,1,channel)
        se = tf.layers.dense(se, channel//ratio, activation=tf.nn.relu,
                             kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer,
                             name='bottleneck_fc')
        assert se.get_shape()[1:] == (1,1,channel//ratio)
        se = tf.layers.dense(se, channel, activation=tf.nn.sigmoid,
                             kernel_initializer=weight_initializer,
                             bias_initializer=bias_initializer,
                             name='recover_fc')
        assert se.get_shape()[1:] == (1,1,channel)
        top = bottom * se
    return top


def residual_block(bottom, params, training, use_seblock=False):
    # first layer (conv-bn-relu)
    top = conv_layer(bottom, params[0], training)

    # second layer (conv-bn-res-relu)
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    top = tf.layers.conv2d(top,
                           filters=params[1][0],
                           kernel_size=params[1][1],
                           strides=(1,1),
                           padding=params[1][2],
                           activation=None,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           name=params[1][3])
    top = norm_layer(top, training, params[1][3]+'/batch_norm')
    if use_seblock:
        top = se_block(top, params[1][3] + '/se_block')
    top += bottom
    top =  tf.nn.relu(top, name=params[1][3] + '/relu')
    return top


def res_identity_block(filters, training, use_bn):
    """identity_block for resnet"""
    def _res_identity_block(bottom):
        # conv 1x1
        path_1 = conv_layer(filters[0], 1)(bottom)
        path_1 = norm_layer(path_1, training, use_bn)
        path_1 = relu(path_1)

        # conv 3x3
        path_1 = conv_layer(filters[1], 3)(path_1)
        path_1 = norm_layer(path_1, training, use_bn)
        path_1 = relu(path_1)

        # conv 1x1
        path_1 = conv_layer(filters[2], 1)(path_1)
        path_1 = norm_layer(path_1, training, use_bn)

        # shortcut
        top = path_1 + bottom
        top = relu(top)

        return top

    return _res_identity_block


def res_conv_block(filters, training, use_bn, strides=2):
    """conv_block for resnet"""
    def _res_conv_block(bottom):
        # shortcut
        path_2 = conv_layer(filters[2], 1, strides=strides)(bottom)
        path_2 = norm_layer(path_2, training, use_bn)

        # conv 1x1
        path_1 = conv_layer(filters[0], 1)(bottom)
        path_1 = norm_layer(path_1, training, use_bn)
        path_1 = relu(path_1)   # activation?

        # conv 3x3
        path_1 = conv_layer(filters[1], 3, strides=strides)(path_1)
        path_1 = norm_layer(path_1, training, use_bn)
        path_1 = relu(path_1)

        # conv 1x1
        path_1 = conv_layer(filters[2], 1)(path_1)
        path_1 = norm_layer(path_1, training, use_bn)

        top = path_1 + path_2
        top = relu(top)
        return top

    return _res_conv_block

def res_block(bottom, filters, training, use_bn, strides=1, downsample=False):

    path_2 = bottom

    # conv 1x1
    path_1 = conv_layer(bottom, filters[0], kernel_size=1)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)   # activation?

    # conv 3x3
    path_1 = conv_layer(path_1, filters[1], kernel_size=3, strides=strides)
    path_1 = norm_layer(path_1, training, use_bn)
    path_1 = relu(path_1)

    # conv 1x1
    path_1 = conv_layer(path_1, filters[2], kernel_size=1)
    path_1 = norm_layer(path_1, training, use_bn)

    if downsample:
        # shortcut
        path_2 = conv_layer(path_2, filters[2], kernel_size=1, strides=strides)
        path_2 = norm_layer(path_2, training, use_bn)

    top = path_1 + path_2
    top = relu(top)
    return top


def conv_layer(bottom, filters, kernel_size, name=None,
               strides=1, padding='same', use_bias=False, kernel_initializer=None):
    """Build a convolutional layer using entry from layer_params)"""
    if kernel_initializer is None:
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()

    if strides is not 1:
        padding = 'valid'
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        bottom = tf.pad(bottom, [[0, 0], [pad_beg, pad_end],
                                 [pad_beg, pad_end], [0, 0]])

    bias_initializer = tf.constant_initializer(value=0.0)

    top = tf.layers.conv2d(bottom,
                           filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           use_bias=use_bias,
                           name=name)
    return top


def pool_layer(bottom, pool, stride, name=None, padding='same'):
    """Short function to build a pooling layer with less syntax"""
    top = tf.layers.max_pooling2d( bottom, pool, stride,
                                   padding=padding,
                                   name=name)
    return top


def relu(bottom, name=None):
    """ Relu actication Function"""
    top = tf.nn.relu(bottom, name=name)
    return top

def norm_layer(bottom, training, use_bn):
    if use_bn:
        top = tf.layers.batch_normalization( bottom, axis=3, 
                                            training=training)
    else:
        top = tf.contrib.layers.group_norm(bottom, groups=32, channels_axis=3)

    return top


def upsampling(bottom, size, name=None):
    """Bilinear Upsampling"""

    out_shape = tf.shape(bottom)[1:3] * tf.constant(size)
    top = tf.image.resize_bilinear(bottom, out_shape, align_corners=True, name=name)
    return top

