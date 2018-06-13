# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)
import logging
import tensorflow as tf
import math
import numpy as np
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module
from tfwrapper import utils

def linear_activation(x):
    '''
    A linear activation function (i.e. no non-linearity)
    '''
    return x

def activation_layer(bottom, name, activation=tf.nn.relu):
    '''
    Custom activation layer, with the tf.nn.relu as default
    '''

    with tf.name_scope(name):

        op = activation(bottom)
        tf.summary.histogram(op.op.name + '/activations', op)

    return op

def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
    strides_aug = [1, strides[0], strides[1], 1]

    op = tf.nn.max_pool(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def max_pool_layer3d(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    kernel_size_aug = [1, kernel_size[0], kernel_size[1], kernel_size[2], 1]
    strides_aug = [1, strides[0], strides[1], strides[2], 1]

    op = tf.nn.max_pool3d(x, ksize=kernel_size_aug, strides=strides_aug, padding=padding)

    return op

def concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''



    return tf.concat(inputs, axis=axis)

def crop_and_concat_layer(inputs, axis=-1):

    '''
    Layer for cropping and stacking feature maps of different size along a different axis. 
    Currently, the first feature map in the inputs list defines the output size. 
    The feature maps can have different numbers of channels. 
    :param inputs: A list of input tensors of the same dimensionality but can have different sizes
    :param axis: Axis along which to concatentate the inputs
    :return: The concatentated feature map tensor
    '''

    output_size = inputs[0].get_shape().as_list()
    concat_inputs = [inputs[0]]

    for ii in range(1,len(inputs)):

        larger_size = inputs[ii].get_shape().as_list()
        start_crop = np.subtract(larger_size, output_size) // 2

        if len(output_size) == 5:  # 3D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], start_crop[3], 0),
                                     (-1, output_size[1], output_size[2], output_size[3], -1))
        elif len(output_size) == 4:  # 2D images
            cropped_tensor = tf.slice(inputs[ii],
                                     (0, start_crop[1], start_crop[2], 0),
                                     (-1, output_size[1], output_size[2], -1))
        else:
            raise ValueError('Unexpected number of dimensions on tensor: %d' % len(output_size))

        concat_inputs.append(cropped_tensor)

    return tf.concat(concat_inputs, axis=axis)


def pad_to_size(bottom, output_size):

    ''' 
    A layer used to pad the tensor bottom to output_size by padding zeros around it
    TODO: implement for 3D data
    '''

    input_size = bottom.get_shape().as_list()
    size_diff = np.subtract(output_size, input_size)

    pad_size = size_diff // 2
    odd_bit = np.mod(size_diff, 2)

    if len(input_size) == 5:
        raise NotImplementedError('This layer has not yet been extended to 3D')

    elif len(input_size) == 4:

        padded =  tf.pad(bottom, paddings=[[0,0],
                                        [pad_size[1], pad_size[1] + odd_bit[1]],
                                        [pad_size[2], pad_size[2] + odd_bit[2]],
                                        [0,0]])

        print('Padded shape:')
        print(padded.get_shape().as_list())


def batch_normalisation_layer(bottom, name, training):
    '''
    Batch normalisation layer (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
    :param bottom: Input layer (should be before activation)
    :param name: A name for the computational graph
    :param training: A tf.bool specifying if the layer is executed at training or testing time 
    :return: Batch normalised activation
    '''

    with tf.variable_scope(name):

        n_out = bottom.get_shape().as_list()[-1]
        tensor_dim = len(bottom.get_shape().as_list())

        if tensor_dim == 2:
            # must be a dense layer
            moments_over_axes = [0]
        elif tensor_dim == 4:
            # must be a 2D conv layer
            moments_over_axes = [0, 1, 2]
        elif tensor_dim == 5:
            # must be a 3D conv layer
            moments_over_axes = [0, 1, 2, 3]
        else:
            # is not likely to be something reasonable
            raise ValueError('Tensor dim %d is not supported by this batch_norm layer' % tensor_dim)

        init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        init_gamma = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
        beta = tf.get_variable(name=name + '_beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
                               trainable=True)
        gamma = tf.get_variable(name=name + '_gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
                                trainable=True)

        batch_mean, batch_var = tf.nn.moments(bottom, moments_over_axes, name=name + '_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(bottom, mean, var, beta, gamma, 1e-3)

        return normed

### FEED_FORWARD LAYERS ##############################################################################33

def conv2D_layer(bottom,
                 name,
                 kernel_size=(3,3),
                 num_filters=32,
                 strides=(1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal'):

    '''
    Standard 2D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv2d(bottom, filter=weights, strides=strides_augm, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def conv3D_layer(bottom,
                 name,
                 kernel_size=(3,3,3),
                 num_filters=32,
                 strides=(1,1,1),
                 activation=tf.nn.relu,
                 padding="SAME",
                 weight_init='he_normal'):

    '''
    Standard 3D convolutional layer
    '''

    bottom_num_filters = bottom.get_shape().as_list()[-1]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv3d(bottom, filter=weights, strides=strides_augm, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def deconv2D_layer(bottom,
                   name,
                   kernel_size=(4,4),
                   num_filters=32,
                   strides=(2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()
    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], num_filters])

    bottom_num_filters = bottom_shape[3]

    weight_shape = [kernel_size[0], kernel_size[1], num_filters, bottom_num_filters]
    bias_shape = [num_filters]
    strides_augm = [1, strides[0], strides[1], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        elif weight_init == 'bilinear':
            weights = _weight_variable_bilinear(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv2d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def deconv3D_layer(bottom,
                   name,
                   kernel_size=(4,4,4),
                   num_filters=32,
                   strides=(2,2,2),
                   output_shape=None,
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):

    '''
    Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a
    factor of 2. 
    '''

    bottom_shape = bottom.get_shape().as_list()

    if output_shape is None:
        output_shape = tf.stack([bottom_shape[0], bottom_shape[1]*strides[0], bottom_shape[2]*strides[1], bottom_shape[3]*strides[2], num_filters])

    bottom_num_filters = bottom_shape[4]

    weight_shape = [kernel_size[0], kernel_size[1], kernel_size[2], num_filters, bottom_num_filters]

    bias_shape = [num_filters]

    strides_augm = [1, strides[0], strides[1], strides[2], 1]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        elif weight_init == 'bilinear':
            weights = _weight_variable_bilinear(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.conv3d_transpose(bottom,
                                    filter=weights,
                                    output_shape=output_shape,
                                    strides=strides_augm,
                                    padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def conv2D_dilated_layer(bottom,
                         name,
                         kernel_size=(3,3),
                         num_filters=32,
                         rate=1,
                         activation=tf.nn.relu,
                         padding="SAME",
                         weight_init='he_normal'):

    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    bottom_num_filters = bottom.get_shape().as_list()[3]

    weight_shape = [kernel_size[0], kernel_size[1], bottom_num_filters, num_filters]
    bias_shape = [num_filters]

    with tf.variable_scope(name):

        if weight_init == 'he_normal':
            N = utils.get_rhs_dim(bottom)
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.nn.atrous_conv2d(bottom, filters=weights, rate=rate, padding=padding)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op


def dense_layer(bottom,
                name,
                hidden_units=512,
                activation=tf.nn.relu,
                weight_init='he_normal'):

    '''
    Dense a.k.a. fully connected layer
    '''

    bottom_flat = utils.flatten(bottom)
    bottom_rhs_dim = utils.get_rhs_dim(bottom_flat)

    weight_shape = [bottom_rhs_dim, hidden_units]
    bias_shape = [hidden_units]

    with tf.name_scope(name):

        if weight_init == 'he_normal':
            N = bottom_rhs_dim
            weights = _weight_variable_he_normal(weight_shape, N, name=name + '_w')
        elif weight_init =='simple':
            weights = _weight_variable_simple(weight_shape, name=name + '_w')
        else:
            raise ValueError('Unknown weight initialisation method %s' % weight_init)

        biases = _bias_variable(bias_shape, name=name + '_b')

        op = tf.matmul(bottom_flat, weights)
        op = tf.nn.bias_add(op, biases)
        op = activation(op)

        # Tensorboard variables
        tf.summary.histogram(weights.name, weights)
        tf.summary.histogram(biases.name, biases)
        tf.summary.histogram(op.op.name + '/activations', op)

        return op

def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)
### BATCH_NORM SHORTCUTS #####################################################################################
def residual_block_bn(bottom,
                    name,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool)):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    temp = batch_normalisation_layer(bottom, name + '_bn_a', training)
    relu = activation(temp)
    conv = conv2D_layer(bottom=relu,
                        name=name+'_a',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)

    temp = batch_normalisation_layer(conv, name + '_bn_b', training)
    relu = activation(temp)
    conv = conv2D_layer(bottom=relu,
                        name=name+'_b',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)


    upscaled = conv2D_layer(bottom=bottom,
                    name=name+'_u',
                    kernel_size=(1,1),
                    num_filters=num_filters,
                    strides=(1,1),
                    activation=linear_activation,
                    padding=padding,
                    weight_init=weight_init)
    
    
    if padding =='VALID':
        logging.info("VAAALIDD")
        return conv + tf.slice(upscaled,[0,2,2,0],[-1,tf.shape(conv)[1],tf.shape(conv)[2],-1])
    else:   
        return conv + upscaled



def conv2D_layer_bn(bottom,
                    name,
                    kernel_size=(3,3),
                    num_filters=32,
                    strides=(1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool)):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    relu = activation(conv_bn)

    return relu


def conv3D_layer_bn(bottom,
                    name,
                    kernel_size=(3,3,3),
                    num_filters=32,
                    strides=(1,1,1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal',
                    training=tf.constant(False, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=linear_activation,
                        padding=padding,
                        weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    relu = activation(conv_bn)

    return relu

def deconv2D_layer_bn(bottom,
                      name,
                      kernel_size=(4,4),
                      num_filters=32,
                      strides=(2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal',
                      training=tf.constant(True, dtype=tf.bool)):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=linear_activation,
                          padding=padding,
                          weight_init=weight_init)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    relu = activation(deco_bn)

    return relu


def deconv3D_layer_bn(bottom,
                      name,
                      kernel_size=(4,4,4),
                      num_filters=32,
                      strides=(2,2,2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal',
                      training=tf.constant(True, dtype=tf.bool),
                      **kwargs):

    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=linear_activation,
                          padding=padding,
                          weight_init=weight_init)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    relu = activation(deco_bn)

    return relu


def conv2D_dilated_layer_bn(bottom,
                           name,
                           kernel_size=(3,3),
                           num_filters=32,
                           rate=1,
                           activation=tf.nn.relu,
                           padding="SAME",
                           weight_init='he_normal',
                           training=tf.constant(True, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=linear_activation,
                                padding=padding,
                                weight_init=weight_init)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    relu = activation(conv_bn)

    return relu



def dense_layer_bn(bottom,
                   name,
                   hidden_units=512,
                   activation=tf.nn.relu,
                   weight_init='he_normal',
                   training=tf.constant(True, dtype=tf.bool)):

    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=linear_activation,
                         weight_init=weight_init)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training)
    relu = activation(batchnorm)

    return relu

def my_crfrnn(unary,rgb_g,image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,name,batch_size):
#    shape=[num_classes,num_classes]
#    tf.initializers.random_uniform
#    
#    shape_unary = tf.shape(unary)
#    shape_rgb = tf.shape(rgb)
#    print(str(shape_unary) + " : " + str(shape_rgb))
    with tf.variable_scope("crf_scope"):
        weights = {
          'spatial_ker_weights': tf.get_variable('spatial_ker_weights',shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform),
          'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights', shape=[num_classes,num_classes],initializer=tf.initializers.random_uniform),
          'compatibility_matrix': tf.get_variable('compatibility_matrix', initializer=np.float32((np.ones((num_classes,num_classes))-np.eye(num_classes))))
          
        }
        
        tf.add_to_collection('crf_weights', weights['spatial_ker_weights'])
        tf.add_to_collection('crf_weights', weights['compatibility_matrix'])
        tf.add_to_collection('crf_weights', weights['bilateral_ker_weights'])
        
      
        
        for batch in range(batch_size):
        
            unaries = tf.transpose(unary[batch, :, :, :], perm=(2, 0, 1))
            rgb = tf.transpose(rgb_g[batch, :, :, :], perm=(2, 0, 1))
        
            c, h, w = num_classes, image_dims[0], image_dims[1]
            all_ones = np.ones((c, h, w), dtype=np.float32)
        
            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=theta_gamma)
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=theta_alpha,
                                                                theta_beta=theta_beta)
            q_values = unaries
        
            for i in range(num_iterations):
                softmax_out = tf.nn.softmax(q_values, dim=0)
        
                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                            theta_gamma=theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
         
                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                              theta_alpha=theta_alpha,
                                                              theta_beta=theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals
        
                # Weighting filter outputs
                message_passing = (tf.matmul(weights["spatial_ker_weights"],
                                             tf.reshape(spatial_out, (c, -1))) +
                                   tf.matmul(weights["bilateral_ker_weights"],
                                             tf.reshape(bilateral_out, (c, -1))))
        
                # Compatibility transform
                pairwise = tf.matmul(weights["compatibility_matrix"], message_passing)
        
                # Adding unary potentials
                pairwise = tf.reshape(pairwise, (c, h, w))
                q_values = unaries - pairwise
          
            if batch == 0:
                output = tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
                
      
            else:
              
                output = tf.concat([output,tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))],0)
                
        return output
    
def my_fixed_crfrnn(unary,rgb_g,image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,name,batch_size):
#    shape=[num_classes,num_classes]
#    tf.initializers.random_uniform
#    
#    shape_unary = tf.shape(unary)
#    shape_rgb = tf.shape(rgb)
#    print(str(shape_unary) + " : " + str(shape_rgb))
    with tf.variable_scope("crf_scope"):
        weights = {
          'spatial_ker_weights': tf.get_variable('spatial_ker_weights',shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform,trainable=False),
          'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights', shape=[num_classes,num_classes],initializer=tf.initializers.random_uniform,trainable=False),
          'compatibility_matrix': tf.get_variable('compatibility_matrix', initializer=np.float32((np.ones((num_classes,num_classes))-np.eye(num_classes))),trainable=False)
          
        }
        
        tf.add_to_collection('crf_weights', weights['spatial_ker_weights'])
        tf.add_to_collection('crf_weights', weights['compatibility_matrix'])
        tf.add_to_collection('crf_weights', weights['bilateral_ker_weights'])
        
      
        
        for batch in range(batch_size):
        
            unaries = tf.transpose(unary[batch, :, :, :], perm=(2, 0, 1))
            rgb = tf.transpose(rgb_g[batch, :, :, :], perm=(2, 0, 1))
        
            c, h, w = num_classes, image_dims[0], image_dims[1]
            all_ones = np.ones((c, h, w), dtype=np.float32)
        
            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=theta_gamma)
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=theta_alpha,
                                                                theta_beta=theta_beta)
            q_values = unaries
        
            for i in range(num_iterations):
                softmax_out = tf.nn.softmax(q_values, dim=0)
        
                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                            theta_gamma=theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
         
                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                              theta_alpha=theta_alpha,
                                                              theta_beta=theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals
        
                # Weighting filter outputs
                message_passing = (tf.matmul(weights["spatial_ker_weights"],
                                             tf.reshape(spatial_out, (c, -1))) +
                                   tf.matmul(weights["bilateral_ker_weights"],
                                             tf.reshape(bilateral_out, (c, -1))))
        
                # Compatibility transform
                pairwise = tf.matmul(weights["compatibility_matrix"], message_passing)
        
                # Adding unary potentials
                pairwise = tf.reshape(pairwise, (c, h, w))
                q_values = unaries - pairwise
          
            if batch == 0:
                output = tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
                
      
            else:
              
                output = tf.concat([output,tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))],0)
                
        return output
    

    
def my_twisted_crfrnn(unary,rgb_g,image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,name,batch_size):
#    shape=[num_classes,num_classes]
#    tf.initializers.random_uniform
#    
#    shape_unary = tf.shape(unary)
#    shape_rgb = tf.shape(rgb)
#    print(str(shape_unary) + " : " + str(shape_rgb))
    with tf.variable_scope("crf_scope"):
        weights = {
          'spa_constant' : tf.constant(np.ones((num_classes,num_classes)),name='spa_constant',dtype=tf.float32),
          'spatial_ker_weights': tf.get_variable('spatial_ker_weights',shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform),
          'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights', shape=[num_classes,num_classes],initializer=tf.initializers.random_uniform),
          'compatibility_matrix': tf.get_variable('compatibility_matrix', initializer=np.float32((np.ones((num_classes,num_classes))-np.eye(num_classes))))
          
        }
        
        tf.add_to_collection('crf_weights', weights['spatial_ker_weights'])
        tf.add_to_collection('crf_weights', weights['compatibility_matrix'])
        tf.add_to_collection('crf_weights', weights['bilateral_ker_weights'])
        
      
        
        for batch in range(batch_size):
        
            unaries = tf.transpose(unary[batch, :, :, :], perm=(2, 0, 1))
            rgb = tf.transpose(rgb_g[batch, :, :, :], perm=(2, 0, 1))
        
            c, h, w = num_classes, image_dims[0], image_dims[1]
            all_ones = np.ones((c, h, w), dtype=np.float32)
        
            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=theta_gamma)
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=theta_alpha,
                                                                theta_beta=theta_beta)
            q_values = unaries
        
            for i in range(num_iterations):
                softmax_out = tf.nn.softmax(q_values, dim=0)
        
                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                            theta_gamma=theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
         
                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                              theta_alpha=theta_alpha,
                                                              theta_beta=theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals
        
                # Weighting filter outputs
                message_passing = (tf.matmul(weights['spa_constant'] - weights["spatial_ker_weights"],
                                             tf.reshape(spatial_out, (c, -1))) +
                                   tf.matmul(weights["bilateral_ker_weights"],
                                             tf.reshape(bilateral_out, (c, -1))))
        
                # Compatibility transform
                pairwise = tf.matmul(weights["compatibility_matrix"], message_passing)
        
                # Adding unary potentials
                pairwise = tf.reshape(pairwise, (c, h, w))
                q_values = unaries - pairwise
          
            if batch == 0:
                output = tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
                
      
            else:
              
                output = tf.concat([output,tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))],0)
                
        return output
    
def diagonal_crfrnn(unary,rgb_g,image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,name,batch_size):
#    shape=[num_classes,num_classes]
#    tf.initializers.random_uniform
#    
#    shape_unary = tf.shape(unary)
#    shape_rgb = tf.shape(rgb)
#    print(str(shape_unary) + " : " + str(shape_rgb))
    print("Num classes : " + str(num_classes))
    with tf.variable_scope("crf_scope"):
        
        diag_init = np.eye(num_classes)
        diag_init[0,0] = 0
        diag_init[num_classes-1,num_classes-1] = 0
        weights = {
          'spa_constant' : tf.constant(10*diag_init,name='spa_constant',dtype=tf.float32),
#          'spatial_ker_weights': tf.get_variable('spatial_ker_weights',shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform),
          'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights', shape=[num_classes,num_classes],initializer=tf.initializers.random_uniform),
          'compatibility_matrix': tf.get_variable('compatibility_matrix', initializer=np.float32((np.ones((num_classes,num_classes))-np.eye(num_classes))))
          
        }
        
        spatial_weights = []
        for m in range(num_classes*num_classes):
            ini = np.float32(np.random.rand(1,1))
            if (((m % (num_classes+1) == 0) & (m != 0) & (m != (num_classes*num_classes-1)) )):
                
                spatial_weights.append(tf.get_variable('spatial_weight_'+str(m), initializer=(ini)))
            else:
                spatial_weights.append(tf.get_variable('spatial_weight_'+str(m), initializer=(-ini)))
        
        spatial_rows = []
        for t in range(num_classes):
            spatial_rows.append(tf.concat([spatial_weights[num_classes*t],spatial_weights[num_classes*t+1],spatial_weights[num_classes*t+2],spatial_weights[num_classes*t+3]], 0))
 
#            to_append = []
#            for k in range(num_classes):
#                to_append.append(spatial_weights[num_classes*t+k])
#            
#            
#            spatial_rows.append(tf.concat(to_append,0))
        
        logging.info(str(spatial_rows))       
                
#            spatial_rows.append(tf.concat([spatial_weights[num_classes*t],spatial_weights[num_classes*t+1],spatial_weights[*t+2],spatial_weights[5*t+3],spatial_weights[5*t+4]], 0))
#            
#        to_rows = []
#        for k in range(num_classes):
#            to_rows.append(spatial_rows[k])
#            
#        spatial_matrix = tf.concat(to_rows,1)
        spatial_matrix = tf.concat([spatial_rows[0],spatial_rows[1],spatial_rows[2],spatial_rows[3]], 1)
        
        tf.add_to_collection('crf_weights', spatial_matrix)
        tf.add_to_collection('crf_weights', weights['compatibility_matrix'])
        tf.add_to_collection('crf_weights', weights['spa_constant'])
        tf.add_to_collection('crf_weights', weights['bilateral_ker_weights'])
        
      
        
        for batch in range(batch_size):
        
            unaries = tf.transpose(unary[batch, :, :, :], perm=(2, 0, 1))
            rgb = tf.transpose(rgb_g[batch, :, :, :], perm=(2, 0, 1))
        
            c, h, w = num_classes, image_dims[0], image_dims[1]
            all_ones = np.ones((c, h, w), dtype=np.float32)
        
            # Prepare filter normalization coefficients
            spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                              theta_gamma=theta_gamma)
            bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                                theta_alpha=theta_alpha,
                                                                theta_beta=theta_beta)
            q_values = unaries
        
            for i in range(num_iterations):
                softmax_out = tf.nn.softmax(q_values, dim=0)
        
                # Spatial filtering
                spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                            theta_gamma=theta_gamma)
                spatial_out = spatial_out / spatial_norm_vals
         
                # Bilateral filtering
                bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                              theta_alpha=theta_alpha,
                                                              theta_beta=theta_beta)
                bilateral_out = bilateral_out / bilateral_norm_vals
        
                # Weighting filter outputs
                message_passing = (tf.matmul(weights['spa_constant'] - spatial_matrix,
                                             tf.reshape(spatial_out, (c, -1))) +
                                   tf.matmul(weights["bilateral_ker_weights"],
                                             tf.reshape(bilateral_out, (c, -1))))
        
                # Compatibility transform
                pairwise = tf.matmul(weights["compatibility_matrix"], message_passing)
        
                # Adding unary potentials
                pairwise = tf.reshape(pairwise, (c, h, w))
                q_values = unaries - pairwise
          
            if batch == 0:
                output = tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))
                
      
            else:
              
                output = tf.concat([output,tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))],0)
                
        return output
### VARIABLE INITIALISERS ####################################################################################

def _weight_variable_simple(shape, stddev=0.02, name=None):

    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        weight = tf.Variable(initial)
    else:
        weight = tf.get_variable(name, initializer=initial)

    tf.add_to_collection('weight_variables', weight)

    return weight

def _weight_variable_he_normal(shape, N, name=None):

    stddev = math.sqrt(2.0/float(N))

    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        weight = tf.Variable(initial)
    else:
        weight = tf.get_variable(name, initializer=initial)

    tf.add_to_collection('weight_variables', weight)

    return weight


def _bias_variable(shape, name=None, init_value=0.0):

    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def _weight_variable_bilinear(shape, name=None):
    '''
    Initialise weights with a billinear interpolation filter for upsampling
    '''

    weights = _bilinear_upsample_weights(shape)
    initial = tf.constant(weights, shape=shape, dtype=tf.float32)

    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def _upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights