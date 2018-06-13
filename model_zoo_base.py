# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import tensorflow as tf
from tfwrapper import layers
import logging
def prostate_base_dropout_two(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=16, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=16, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=32, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=64, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
        pool3=tf.nn.dropout(pool3,keep_prob)
        
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=128, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training, padding='VALID')
        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training, padding='VALID')
        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training, padding='VALID')
        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=32, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=16, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        net_out = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])
        
    pred = layers.my_crfrnn(net_out,images,image_dims=(320,320),
         num_classes=nlabels,
         theta_alpha=160.,
         theta_beta=3,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    return net_out,pred


def prostate_base_dropout(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=16, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=16, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=32, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=64, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
        pool3=tf.nn.dropout(pool3,keep_prob)
        
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=128, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training, padding='VALID')
        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training, padding='VALID')
        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training, padding='VALID')
        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=32, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=16, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])
    return pred
 
def prostate_base_old(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=16, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=16, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=32, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=64, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=128, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=32, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=16, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])


    return pred

       
def prostate_base_network(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=16, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=16, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=32, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=64, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=128, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=256, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=128, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=64, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=32, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=16, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])

    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
         num_classes=nlabels,
         theta_alpha=160.,
         theta_beta=3,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    return pred

def prostate_diagonal(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=16, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
    pred = layers.diagonal_crfrnn(pred,images,image_dims=(320,320),
             num_classes=nlabels,
             theta_alpha=160.,
             theta_beta=3,
             theta_gamma=10.,
             num_iterations=5,
             name='crfrnn',batch_size=4)
    return pred

def prostate_deep_base(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=32, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=64, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=128, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=256, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=512, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=256, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=128, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=64, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=32, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        

    return pred


        
        
def prostate_deep_rnn(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=32, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=64, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=128, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=256, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=512, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=256, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=128, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=64, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=32, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
             num_classes=nlabels,
             theta_alpha=250.,
             theta_beta=3,
             theta_gamma=10.,
             num_iterations=5,
             name='crfrnn',batch_size=4)
    return pred


def prostate_new_crf(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=16, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
             num_classes=nlabels,
             theta_alpha=250.,
             theta_beta=3,
             theta_gamma=10.,
             num_iterations=5,
             name='crfrnn',batch_size=4)
    return pred

def prostate_dropout_two(images, training, nlabels,keep_prob):


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=16, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        
        pool3=tf.nn.dropout(pool3,keep_prob)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='SAME')

        net_out = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
    pred = layers.my_crfrnn(net_out,images,image_dims=(320,320),
             num_classes=nlabels,
             theta_alpha=250.,
             theta_beta=3,
             theta_gamma=10.,
             num_iterations=5,
             name='crfrnn',batch_size=4)
    return net_out,pred

def prostate_deep_dropout(images, training, nlabels,keep_prob):


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=32, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=64, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=128, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        
        pool3=tf.nn.dropout(pool3,keep_prob)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=256, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=512, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=256, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=128, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=64, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=32, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

    return pred


def prostate_deep_dropout_rnn(images, training, nlabels,keep_prob):


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=32, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=64, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=128, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        
        pool3=tf.nn.dropout(pool3,keep_prob)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=256, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=512, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=256, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=128, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=64, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=32, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
         num_classes=nlabels,
         theta_alpha=250.,
         theta_beta=3,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    return pred


def prostate_dropout(images, training, nlabels,keep_prob):


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=16, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        
        pool3=tf.nn.dropout(pool3,keep_prob)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

    return pred


def prostate_base_new(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=16, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=32, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=64, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=128, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=256, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=128, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=64, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=32, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=16, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

    return pred
def diagonal_network(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
    
        
    pred = layers.diagonal_crfrnn(pred,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return pred


def twisted_network(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
    
        
    pred = layers.my_twisted_crfrnn(pred,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=1.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return pred

#def prostate_base_new(images, training, nlabels):
#
#
#    with tf.Session() as session:
#        print(str(session.run(tf.shape(images))))
#        print(str(session.run(tf.shape(images))))
#
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
#        conv1_2 = layers.residual_block_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='SAME')
#    
#        pool1 = layers.max_pool_layer2d(conv1_2)
#    
#        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
#        conv2_2 = layers.residual_block_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='SAME')
#    
#        pool2 = layers.max_pool_layer2d(conv2_2)
#    
#        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
#        conv3_2 = layers.residual_block_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='SAME')
#    
#        pool3 = layers.max_pool_layer2d(conv3_2)
#    
#        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
#        
#        pool4 = layers.max_pool_layer2d(conv4_1)
#    
#        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
#        
#        
#        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
#        
#        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
#    
#        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
#        
#        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
#        
#        concat3 = layers.concat_layer([upconv3, conv3_2], axis=3)
#    
#        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')
#        conv7_2 = layers.residual_block_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='SAME')
#    
#    
#        conv7_3 = layers.residual_block_bn(conv7_2, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
#        
#        concat2 = layers.concat_layer([upconv2, conv2_2], axis=3)
#    
#        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')
#        conv8_2 = layers.residual_block_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='SAME')
#    
#    
#        conv8_3 = layers.residual_block_bn(conv8_2, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
#        
#        
#        concat1 = layers.concat_layer([upconv1, conv1_2], axis=3)
#    
#        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
#        conv9_2 = layers.residual_block_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='SAME')
#    
#        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#
#    return pred




def prostate_full_sup_crf_new(images, training, nlabels):


    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
        conv1_2 = layers.residual_block_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='SAME')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        conv2_2 = layers.residual_block_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='SAME')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
        conv3_2 = layers.residual_block_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='SAME')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')
        conv7_2 = layers.residual_block_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='SAME')
    
    
        conv7_3 = layers.residual_block_bn(conv7_2, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')
        conv8_2 = layers.residual_block_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='SAME')
    
    
        conv8_3 = layers.residual_block_bn(conv8_2, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
        conv9_2 = layers.residual_block_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='SAME')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

        with tf.Session() as session:
            logging.info(str(session.run(tf.shape(pred))))

        pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
             num_classes=3,
             theta_alpha=160.,
             theta_beta=3,
             theta_gamma=10.,
             num_iterations=5,
             name='crfrnn',batch_size=4)
    return pred

#def prostate_base_network(images, training, nlabels):
#
#    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
#
##    with tf.Session() as session:
##        print(str(session.run(tf.shape(images))))
##        print(str(session.run(tf.shape(images_padded))))
#
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
#        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
#    
#        pool1 = layers.max_pool_layer2d(conv1_2)
#    
#        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
#        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
#    
#        pool2 = layers.max_pool_layer2d(conv2_2)
#    
#        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
#        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
#    
#        pool3 = layers.max_pool_layer2d(conv3_2)
#    
#        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
#        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
#    
#        pool4 = layers.max_pool_layer2d(conv4_2)
#    
#        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
#        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
#    
#        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
#    
#        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
#        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
#    
#        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    
#        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
#    
#        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
#        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
#    
#        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
#    
#        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
#        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
#    
#        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
#    
#        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
#        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
#    
#        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#        
#        pred = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])
##        with tf.Session() as session:
##            print(str(session.run(tf.shape(pred))))
##    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
##                 num_classes=4,
##                 theta_alpha=1.,
##                 theta_beta=0.01,
##                 theta_gamma=30.,
##                 num_iterations=5,
##                 name='crfrnn',batch_size=4)
#
#    return pred


def prostate_crf_network(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.Session() as session:
        print(str(session.run(tf.shape(images))))
        print(str(session.run(tf.shape(images_padded))))


    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred = tf.slice(pred,[0,2,2,0],[-1,320,320,-1])
        with tf.Session() as session:
            print(str(session.run(tf.shape(pred))))
    pred = layers.my_crfrnn(pred,images,image_dims=(320,320),
                 num_classes=4,
                 theta_alpha=160.,
                 theta_beta=3,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)

    return pred

#def unet2D_bn_modified(images, training, nlabels):
#
#    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
#        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
#    
#        pool1 = layers.max_pool_layer2d(conv1_2)
#    
#        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
#        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
#    
#        pool2 = layers.max_pool_layer2d(conv2_2)
#    
#        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
#        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
#    
#        pool3 = layers.max_pool_layer2d(conv3_2)
#    
#        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
#        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
#    
#        pool4 = layers.max_pool_layer2d(conv4_2)
#    
#        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
#        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
#    
#        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
#    
#        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
#        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
#    
#        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    
#        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
#    
#        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
#        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
#    
#        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
#    
#        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
#        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
#    
#        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
#    
#        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
#        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
#    
#        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#        
#        
#        
#    pred = layers.my_crfrnn(network_output,images,image_dims=(212, 212),
#                 num_classes=5,
#                 theta_alpha=160.,
#                 theta_beta=3.,
#                 theta_gamma=10.,
#                 num_iterations=5,
#                 name='crfrnn',batch_size=4)
# 
#    return pred


#def residual_rnn(images, training, nlabels):
#
#    images_padded = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.residual_block_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='SAME')
#        
#        pool1 = layers.max_pool_layer2d(conv1_1)
#    
#        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
#        
#        pool2 = layers.max_pool_layer2d(conv2_1)
#    
#        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
# 
#        pool3 = layers.max_pool_layer2d(conv3_1)
#    
#        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
#        
#        pool4 = layers.max_pool_layer2d(conv4_1)
#    
#        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
#        
#        
#        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
#        
#        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
#    
#        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
#        
#        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
#        
#        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
#    
#        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')
#
#    
#        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
#        
#        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
#    
#        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')
#
#        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
#        
#        
#        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
#    
#        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
#
#        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
##        
##        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
#        
#        
#    pred = layers.my_crfrnn(pred,images_padded,image_dims=(224, 224),
#         num_classes=5,
#         theta_alpha=160.,
#         theta_beta=3.,
#         theta_gamma=10.,
#         num_iterations=5,
#         name='crfrnn',batch_size=4)
#    
#    
#    return tf.slice(pred,[0,6,6,0],[-1,212,212,-1])


def residual_rnn(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
    
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
    
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='VALID')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, tf.slice(conv1_1,[0,4,4,0],[-1,216,216,-1])], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
#        
#        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
        
        
    pred = layers.my_crfrnn(pred,images,image_dims=(212, 212),
         num_classes=5,
         theta_alpha=250.,
         theta_beta=3.,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    
    
    return pred


def residual_dropout_rnn(images, training, nlabels, keep_prob):

    images_padded = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
    
    pred = layers.my_crfrnn(pred,images,image_dims=(212, 212),
         num_classes=5,
         theta_alpha=160.,
         theta_beta=3.,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    
    return pred

def residual_dropout(images, training, nlabels, keep_prob):

    images = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
    return pred


#def unet2D_bn_modified(images, training, nlabels):
#
#    images = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=32, training=training, padding='SAME')
#        
#        pool1 = layers.max_pool_layer2d(conv1_1)
#    
#        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=64, training=training, padding='SAME')
#        
#        pool2 = layers.max_pool_layer2d(conv2_1)
#    
#        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=128, training=training, padding='SAME')
# 
#        pool3 = layers.max_pool_layer2d(conv3_1)
#    
#        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=256, training=training, padding='SAME')
#        
#        pool4 = layers.max_pool_layer2d(conv4_1)
#    
#        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=512, training=training, padding='SAME')
#        
#        
#        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
#        
#        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
#    
#        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=256, training=training, padding='SAME')
#        
#        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
#        
#        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
#    
#        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=128, training=training, padding='SAME')
#
#    
#        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
#        
#        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
#    
#        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=64, training=training, padding='SAME')
#
#        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
#        
#        
#        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
#    
#        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=32, training=training, padding='SAME')
#
#        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#        
#        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
#    return pred

#def unet2D_bn_modified(images, training, nlabels):
#
#    images = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.residual_block_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
#        
#        pool1 = layers.max_pool_layer2d(conv1_1)
#    
#        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
#        
#        pool2 = layers.max_pool_layer2d(conv2_1)
#    
#        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
# 
#        pool3 = layers.max_pool_layer2d(conv3_1)
#    
#        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
#        
#        pool4 = layers.max_pool_layer2d(conv4_1)
#    
#        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
#        
#        
#        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
#        
#        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
#    
#        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
#        
#        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
#        
#        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
#    
#        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')
#
#    
#        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
#        
#        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
#    
#        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')
#
#        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
#    
#        
#        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
#        
#        
#        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
#    
#        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
#
#        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#        
#        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
#    return pred

def unet2D_bn_two_outputs(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
    
        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        
        
    pred = layers.my_crfrnn(network_output,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return network_output,pred

def unet2D_bn_dropout_two_outputs(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        
        
    pred = layers.my_crfrnn(network_output,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return network_output,pred

#def seperate_net(images, training, nlabels,keep_prob):
#    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
#
#    with tf.variable_scope("network_scope"):
#        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
#        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
#    
#        pool1 = layers.max_pool_layer2d(conv1_2)
#    
#        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
#        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
#    
#        pool2 = layers.max_pool_layer2d(conv2_2)
#    
#        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
#        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
#    
#        pool3 = layers.max_pool_layer2d(conv3_2)
#        pool3=tf.nn.dropout(pool3,keep_prob)
#        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
#        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
#    
#        pool4 = layers.max_pool_layer2d(conv4_2)
#        pool4=tf.nn.dropout(pool4,keep_prob)
#        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
#        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
#    
#        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
#        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
#    
#        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
#        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
#        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
#        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    
#        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
#    
#        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
#        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
#        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
#        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
#    
#        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
#        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
#    
#        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
#    
#        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
#        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
#
#        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#        
#        return network_output

def seperate_non_net(images, training, nlabels):
    images_padded = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
 
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)

        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')

        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    

        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')

        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
    return pred
    

def seperate_net(images, training, nlabels,keep_prob):
    images_padded = tf.pad(images, [[0,0], [6, 6], [6, 6], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.residual_block_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='SAME')
        
        pool1 = layers.max_pool_layer2d(conv1_1)
    
        conv2_1 = layers.residual_block_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
        
        pool2 = layers.max_pool_layer2d(conv2_1)
    
        conv3_1 = layers.residual_block_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
 
        pool3 = layers.max_pool_layer2d(conv3_1)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.residual_block_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
        
        pool4 = layers.max_pool_layer2d(conv4_1)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.residual_block_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
        
        
        conv5_3 = layers.residual_block_bn(conv5_1, 'conv5_2', num_filters=4*nlabels, training=training, padding='SAME')
        conv5_3=tf.nn.dropout(conv5_3,keep_prob)
        upconv4 = layers.pixelShuffler(conv5_3,scale=2)
        
        concat4 = layers.concat_layer([upconv4, conv4_1], axis=3)
    
        conv6_1 = layers.residual_block_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
        
        conv6_3 = layers.residual_block_bn(conv6_1, 'conv6_2', num_filters=4*nlabels, training=training, padding='SAME')
    
        conv6_3=tf.nn.dropout(conv6_3,keep_prob)
        upconv3 = layers.pixelShuffler(conv6_3,scale=2)
        
        concat3 = layers.concat_layer([upconv3, conv3_1], axis=3)
    
        conv7_1 = layers.residual_block_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')

    
        conv7_3 = layers.residual_block_bn(conv7_1, 'conv7_3', num_filters=4*nlabels, training=training, padding='SAME')
        conv7_3=tf.nn.dropout(conv7_3,keep_prob)
        
        upconv2 = layers.pixelShuffler(conv7_3,scale=2)
        
        concat2 = layers.concat_layer([upconv2, conv2_1], axis=3)
    
        conv8_1 = layers.residual_block_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')

        conv8_3 = layers.residual_block_bn(conv8_1, 'conv8_3', num_filters=4*nlabels, training=training, padding='SAME')
    
        
        upconv1 = layers.pixelShuffler(conv8_3,scale=2)
        
        
        concat1 = layers.concat_layer([upconv1, conv1_1], axis=3)
    
        conv9_1 = layers.residual_block_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')

        pred = layers.conv2D_layer_bn(conv9_1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        pred= tf.slice(pred,[0,6,6,0],[-1,212,212,-1])
    return pred
    
def seperate_crf_rnn(network_output,images,nlabels):
    
    pred = layers.my_crfrnn(network_output,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=1)
    return pred

def seperate_crf_rnn_pro(network_output,images,nlabels):
    
    pred = layers.my_crfrnn(network_output,images,image_dims=(320, 320),
                 num_classes=4,
                 theta_alpha=250.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=1)
    return pred


def heart_rnn_dropout(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        
        
    pred = layers.my_crfrnn(network_output,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return pred

def heart_fixed_rnn_dropout(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
        pool3=tf.nn.dropout(pool3,keep_prob)
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
        pool4=tf.nn.dropout(pool4,keep_prob)
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        conv5_2=tf.nn.dropout(conv5_2,keep_prob)
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
        conv6_2=tf.nn.dropout(conv6_2,keep_prob)
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
        conv7_2=tf.nn.dropout(conv7_2,keep_prob)
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
    
        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

        network_output = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
        
        
        
    pred = layers.my_fixed_crfrnn(network_output,images,image_dims=(212, 212),
                 num_classes=5,
                 theta_alpha=160.,
                 theta_beta=3.,
                 theta_gamma=10.,
                 num_iterations=5,
                 name='crfrnn',batch_size=4)
 
    return pred


def unet2D_bn_dropout(images, training, nlabels,keep_prob):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')

    pool3 = layers.max_pool_layer2d(conv3_2)
    pool3=tf.nn.dropout(pool3,keep_prob)
    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')

    pool4 = layers.max_pool_layer2d(conv4_2)
    pool4=tf.nn.dropout(pool4,keep_prob)
    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')

    conv5_2=tf.nn.dropout(conv5_2,keep_prob)
    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    conv6_2=tf.nn.dropout(conv6_2,keep_prob)
    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    conv7_2=tf.nn.dropout(conv7_2,keep_prob)
    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

    return pred


#def unet2D_bn(images, training, nlabels):
#
#    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
#
#    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
#    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
#
#    pool1 = layers.max_pool_layer2d(conv1_2)
#
#    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
#    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
#
#    pool2 = layers.max_pool_layer2d(conv2_2)
#
#    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
#    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
#
#    pool3 = layers.max_pool_layer2d(conv3_2)
#
#    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
#    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
#
#    pool4 = layers.max_pool_layer2d(conv4_2)
#
#    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
#    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
#
#    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
#
#    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
#    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
#
#    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#
#    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
#
#    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
#    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
#
#    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
#
#    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
#    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
#
#    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
#    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
#
#    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
#    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
#
#    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#
#    return pred

def unet2D_bn_modified(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    with tf.variable_scope("network_scope"):
        conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
    
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
    
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
    
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
    
        pool4 = layers.max_pool_layer2d(conv4_2)
    
        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
    
        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
    
        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
    
        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    
        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
    
        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
    
        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
    
        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
    
        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
    
        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
    pred = layers.my_crfrnn(pred,images,image_dims=(212, 212),
         num_classes=5,
         theta_alpha=160.,
         theta_beta=3.,
         theta_gamma=10.,
         num_iterations=5,
         name='crfrnn',batch_size=4)
    return pred



def unet2D_bn(images, training, nlabels):

    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')

    return pred



#def unet2D_bn(images, training, nlabels):
#
#    images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')
#
#    conv1_1 = layers.conv2D_layer_bn(images_padded, 'conv1_1', num_filters=64, training=training, padding='VALID')
#    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='VALID')
#
#    pool1 = layers.max_pool_layer2d(conv1_2)
#
#    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='VALID')
#    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='VALID')
#
#    pool2 = layers.max_pool_layer2d(conv2_2)
#
#    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='VALID')
#    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='VALID')
#
#    pool3 = layers.max_pool_layer2d(conv3_2)
#
#    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='VALID')
#    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='VALID')
#
#    pool4 = layers.max_pool_layer2d(conv4_2)
#
#    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='VALID')
#    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='VALID')
#
#    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, training=training)
#    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)
#
#    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='VALID')
#    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='VALID')
#
#    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, training=training)
#
#    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)
#
#    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='VALID')
#    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='VALID')
#
#    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, training=training)
#    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)
#
#    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='VALID')
#    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='VALID')
#
#    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, training=training)
#    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)
#
#    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='VALID')
#    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='VALID')
#
#    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=layers.linear_activation, training=training, padding='VALID')
#
#    return pred