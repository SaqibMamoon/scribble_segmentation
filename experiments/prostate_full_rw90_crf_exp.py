# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:57:24 2018

@author: ybara
"""

import model_zoo
import tensorflow as tf

experiment_name = 'prostate_all_full_rw_90_crf'

# Model settings
model_handle = model_zoo.prostate_crf_network

# Data settings
data_mode = '2D'  # 2D or 3D
image_size = (320, 320)
target_resolution = (1.36719, 1.36719)
scribble_data = '/scribble_data/prostate_divided.h5'        #Path from project root

# Training settings
batch_size = 4
learning_rate = 0.0001
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'weighted_crossentropy'  # crossentropy/weighted_crossentropy/dice

# Augmentation settings
augment_batch = False
do_rotations = True
do_scaleaug = False
do_fliplr = False

# Rarely changed settings
use_data_fraction = False  # Should normally be False
nlabels = 4
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100

# Weak supervision settings
random_walk = True
rw_beta = 1000
rw_threshold = 0.90
epochs_per_recursion = 100
number_of_recursions = 4
reinit = False                  # if true, will reinitialise network weights between recursions
keep_largest_cluster = True    # if true, will only keep the largest cluster in the output
cnn_threshold = None              # if defined, will threshold output of CNN such that more unlabelled pixels are present
                                # (therefore more space for random walker + recursion to do work)
rw_intersection = True         # if true, will random walk to fully segment image based off original scribble
                                # then limit output to the regions defined by the low threshold random walk
rw_reversion = True            # if true, will attempt to revert to original random walked scribbles if the
                                # if CNN + postprocessing predicts a smaller structure than was in the original
                                # scribble
                                #Parameters for gaussian edge smoothing
                                #Larger sigma blurs more, smaller threshold results in more growth
edge_smoother_sigma = 2         #
edge_smoother_threshold = 0.7   # between 0 & 1

percent_full_sup = 0
length_ratio = 1         # factor by which to reduce the length of the scribbles

#AUTOCALCULATED
postprocessing = bool(reinit + keep_largest_cluster + bool(cnn_threshold) + rw_intersection + rw_reversion)
max_epochs = number_of_recursions*epochs_per_recursion
smooth_edges = (not edge_smoother_sigma is None) and (not edge_smoother_threshold is None)