# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging

import argparse
import metrics_acdc_simple
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform
import config.system as sys_config
import acdc_data_welch as acdc_data
from acdc_data_welch import most_recent_recursion
import config.system as sys_config
import model_sep_pro as model
import utils
import image_utils
import h5py
from scipy.stats import ttest_ind
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
#sys_config.setup_GPU_environment()
sys_config.setup_GPU_environment()

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def average_outputs(sess,softmax_pl,images_pl,network_input,keep_prob):

    temp = np.zeros((50,1,320,320,4))
    for k in range(50):
        

        temp[k,:,:,:,:] = sess.run(softmax_pl,feed_dict={images_pl: network_input,keep_prob:0.5})

    
    
    
    temp_mean = np.mean(temp,axis=0)
    return temp,temp_mean



def predict_next_gt(data,
                    images_train,
                    images_placeholder,
                    training_time_placeholder,
                    keep_prob,
                    logits,
                    network_output,
                    rnn_in_placeholder,
                    sess):
    '''
    Uses current network weights to segment images for next recursion
    After postprocessing, these are used as the ground truth for further training
    :param data: Data of the current recursion - if this is of recursion n, this function
                 predicts ground truths for recursion n + 1
    :param images_train: Numpy array of training images
    :param images_placeholder: Tensorflow placeholder for image feed
    :param training_time_placeholder: Boolean tensorflow placeholder
    :param logits: Logits operator for calculating segmentation mask probabilites
    :param sess: Tensorflow session
    :return: The data file for recursion n + 1
    '''
    #get recursion from filename
    recursion = utils.get_recursion_from_hdf5(data)

    new_recursion_fname = acdc_data.recursion_filepath(recursion + 1, data_file=data)
    if not os.path.exists(new_recursion_fname):
        fpath = os.path.dirname(data.filename)
        data.close()
        data = acdc_data.create_recursion_dataset(fpath, recursion + 1)
    else:
        data.close()
        data = h5py.File(new_recursion_fname, 'r+')

    #attributes to track processing
    prediction = data['predicted']
    prediction_pre_cor = data['predicted_pre_cor']
    p_vals = data['p_vals']
    processed = data['predicted'].attrs.get('processed')
    if not processed:
        processed_to = data['predicted'].attrs.get('processed_to')
        scr_max = len(images_train)
        print("SCR max = " + str(scr_max))
        for scr_idx in range(processed_to, scr_max):
            ind=np.copy(scr_idx)
 

            x = np.expand_dims(np.expand_dims(np.array(images_train[ind, ...]), -1),0)

            feed_dict = {
                images_placeholder: x,
                training_time_placeholder: False,
                keep_prob:0.5
            }
            

            
            ps = np.zeros((320,320))
            temp,temp_mean = average_outputs(sess,network_output,images_placeholder,x,keep_prob)
            
            logits_out = np.squeeze(sess.run(logits,feed_dict={images_placeholder: x,rnn_in_placeholder:temp_mean}))
   
            conf = 0.05
                

            
            mask = tf.arg_max(logits_out, dimension=-1)      
            mask_out = np.squeeze(sess.run(mask, feed_dict=feed_dict))
            
            pre_cor_mask = np.copy(mask_out)
            
       
#            for m in range(212):
#                for n in range(212):
#      
#                    
#                    estim = mask_out[m,n]
#                     
#                    
#                    temp_mean[0,m,n,estim] = -1000000
#                    
#                    best_ind = np.argmax(np.squeeze(np.copy(temp_mean[0,m,n,:])))
#                    [st,p] = ttest_ind(np.squeeze(temp[:,0,m,n,estim]),np.squeeze(temp[:,0,m,n,best_ind]),equal_var=False)
#                    ps[m,n] = p
#                    if p > conf:
#                        mask_out[m,n] = 0
                        
                    

            prediction_pre_cor[ind, ...] = np.squeeze(pre_cor_mask)
            prediction[ind, ...] = np.squeeze(mask_out)
            p_vals[ind, ...] = np.squeeze(ps)
            data['predicted'].attrs.modify('processed_to', scr_idx + exp_config.batch_size)

        if exp_config.reinit:
            logging.info("Initialising variables")
            sess.run(tf.global_variables_initializer())

        data['predicted'].attrs.modify('processed', True)
        logging.info('Created unprocessed ground truths for recursion {}'.format(recursion + 1))

        #Reopen in read only mode
        data.close()
        data = h5py.File(new_recursion_fname, 'r')
    return data



if __name__ == '__main__':

    

    model_path = '/scratch_net/biwirender02/cany/scribble/logdir/prostate_deep_dropout_rnn_net_mean_exp'
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()
    log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
    # input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    base_data, recursion_data, recursion = acdc_data.load_and_maybe_process_scribbles(scribble_file=sys_config.project_root + exp_config.scribble_data,
                                                                              target_folder=log_dir,
                                                                              percent_full_sup=exp_config.percent_full_sup,
                                                                              scr_ratio=exp_config.length_ratio
                                                                              )
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    net_tensor_shape = [batch_size] + list(exp_config.image_size) + [4]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    keep_prob = tf.placeholder(tf.float32, shape=[])
    rnn_in_placeholder = tf.placeholder(tf.float32, shape=net_tensor_shape, name='images')
    training_time_placeholder = tf.placeholder(tf.bool, shape=[])
    net_out = model.predict_net_out(images_pl, keep_prob,exp_config.model_handle, exp_config.nlabels)
    
    softmax_pl = model.predict_final(images_pl, rnn_in_placeholder, exp_config.nlabels)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess,"/scratch_net/biwirender02/cany/scribble/logdir/prostate_deep_dropout_rnn_net_mean_exp/recursion_0_model_best_dice.ckpt-3999")


        images_train = np.array(base_data['images_train'])
        recursion_data = predict_next_gt(data=recursion_data,
                                     images_train=images_train,
                                     images_placeholder=images_pl,
                                     training_time_placeholder=training_time_placeholder,
                                     keep_prob=keep_prob,
                                     logits=softmax_pl,
                                     network_output=net_out,
                                     rnn_in_placeholder=rnn_in_placeholder,
                                     sess=sess)


