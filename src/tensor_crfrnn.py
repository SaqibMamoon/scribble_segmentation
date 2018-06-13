# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 19:06:15 2018

@author: ybara
"""



import numpy as np
import tensorflow as tf
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module

def my_crfrnn(unary,rgb,image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations,name):
    
    
    weights = {
      'spatial_ker_weights': tf.get_variable('spatial_ker_weights', shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform),
      'bilateral_ker_weights': tf.get_variable('bilateral_ker_weights', shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform),
      'compatibility_matrix': tf.get_variable('compatibility_matrix', shape=[num_classes,num_classes], initializer=tf.initializers.random_uniform)
      
    }
    
    unaries = tf.transpose(unary[0, :, :, :], perm=(2, 0, 1))
    rgb = tf.transpose(rgb[0, :, :, :], perm=(2, 0, 1))

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
        print(tf.shape(spatial_out))
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

    return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

