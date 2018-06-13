# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import image_utils
import model_dropout as model
from background_generator import BackgroundGenerator
import config.system as sys_config
import acdc_data_crf as acdc_data
import random_walker
import h5py
from scipy.ndimage.filters import gaussian_filter
import pydensecrf.densecrf as dcrf
import scipy
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from scipy.stats import ttest_ind
### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:

#from experiments import test as exp_config
from experiments import heart_unet_welch_crf_dif_exp as exp_config

# from experiments import unet3D_bn_modified as exp_config
# from experiments import unet2D_bn_wxent as exp_config
# from experiments import FCN8_bn_wxent as exp_config

########################################################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
#sys_config.setup_GPU_environment()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_crf=False
just_started = True

try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')


def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)
    already_created_recursion = False
    print("ALready created recursion : " + str(already_created_recursion))
    init_step = 0
    # Load data
    base_data, recursion_data, recursion = acdc_data.load_and_maybe_process_scribbles(scribble_file=sys_config.project_root + exp_config.scribble_data,
                                                                                      target_folder='/scratch_net/biwirender02/cany/scribble/logdir/heart_dropout_welch_non_exp',
                                                                                      percent_full_sup=exp_config.percent_full_sup,
                                                                                      scr_ratio=exp_config.length_ratio
                                                                                      )
    #wrap everything from this point onwards in a try-except to catch keyboard interrupt so
    #can control h5py closing data
    try:
        loaded_previous_recursion = False
        start_epoch = 0
        if continue_run:
            logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            try:
                try:
                    init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'recursion_{}_model.ckpt'.format(recursion))
                
                except:
                    print("EXCEPTE GİRDİ")
                    init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir, 'recursion_{}_model.ckpt'.format(recursion - 1))
                    loaded_previous_recursion = True
                logging.info('Checkpoint path: %s' % init_checkpoint_path)
                init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
                start_epoch = int(init_step/(len(base_data['images_train'])/4))
                logging.info('Latest step was: %d' % init_step)
                logging.info('Continuing with epoch: %d' % start_epoch)
            except:
                logging.warning('!!! Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
                continue_run = False
                init_step = 0
                start_epoch = 0

            logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


        if loaded_previous_recursion:
            logging.info("Data file exists for recursion {} "
                         "but checkpoints only present up to recursion {}".format(recursion, recursion - 1))
            logging.info("Likely means postprocessing was terminated")
            
            if not already_created_recursion:
                
                recursion_data = acdc_data.load_different_recursion(recursion_data, -1)
                recursion-=1
            else:
                start_epoch = 0
                init_step = 0
        # load images and validation data
        images_train = np.array(base_data['images_train'])

        # if exp_config.use_data_fraction:
        #     num_images = images_train.shape[0]
        #     new_last_index = int(float(num_images)*exp_config.use_data_fraction)
        #
        #     logging.warning('USING ONLY FRACTION OF DATA!')
        #     logging.warning(' - Number of imgs orig: %d, Number of imgs new: %d' % (num_images, new_last_index))
        #     images_train = images_train[0:new_last_index,...]
        #     labels_train = labels_train[0:new_last_index,...]

        logging.info('Data summary:')
        logging.info(' - Images:')
        logging.info(images_train.shape)
        logging.info(images_train.dtype)
        #logging.info(' - Labels:')
        #logging.info(labels_train.shape)
        #logging.info(labels_train.dtype)

        # Tell TensorFlow that the model will be built into the default Graph.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
#        with tf.Graph().as_default():
        with tf.Session(config = config) as sess:
            # Generate placeholders for the images and labels.

            image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
            mask_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)

            images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
            labels_placeholder = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')

            learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
            training_time_placeholder = tf.placeholder(tf.bool, shape=[])

            keep_prob = tf.placeholder(tf.float32, shape=[])
            tf.summary.scalar('learning_rate', learning_rate_placeholder)

            # Build a Graph that computes predictions from the inference model.
            logits = model.inference(images_placeholder,
                                     keep_prob,
                                     exp_config.model_handle,
                                     training=training_time_placeholder,
                                     nlabels=exp_config.nlabels)

            # Add to the Graph the Ops for loss calculation.
            [loss, _, weights_norm] = model.loss(logits,
                                                 labels_placeholder,
                                                 nlabels=exp_config.nlabels,
                                                 loss_type=exp_config.loss_type,
                                                 weight_decay=exp_config.weight_decay)  # second output is unregularised loss

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('weights_norm_term', weights_norm)


            
            # Add to the Graph the Ops that calculate and apply gradients.
   
          

            # Build the summary Tensor based on the TF collection of Summaries.
            summary = tf.summary.merge_all()

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            # Only keep two checkpoints, as checkpoints are kept for every recursion
            # and they can be 300MB +
            saver = tf.train.Saver(max_to_keep=2)
            saver_best_dice = tf.train.Saver(max_to_keep=2)
            saver_best_xent = tf.train.Saver(max_to_keep=2)

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            # with tf.name_scope('monitoring'):

            val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error')
            val_error_summary = tf.summary.scalar('validation_loss', val_error_)

            val_dice_ = tf.placeholder(tf.float32, shape=[], name='val_dice')
            val_dice_summary = tf.summary.scalar('validation_dice', val_dice_)

            val_summary = tf.summary.merge([val_error_summary, val_dice_summary])

            train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error')
            train_error_summary = tf.summary.scalar('training_loss', train_error_)

            train_dice_ = tf.placeholder(tf.float32, shape=[], name='train_dice')
            train_dice_summary = tf.summary.scalar('training_dice', train_dice_)

            train_summary = tf.summary.merge([train_error_summary, train_dice_summary])

            # Run the Op to initialize the variables.
            sess.run(init)
            


#            if continue_run:
#                # Restore session
#                saver.restore(sess, init_checkpoint_path)
            
#            saver.restore(sess,'/scratch_net/biwirender02/cany/scribble/logdir/heart_residual_crf/recursion_0_model_best_dice.ckpt-14199')
#
#            saver.restore(sess,"/scratch_net/biwirender02/cany/scribble/logdir/"+str(exp_config.experiment_name)+ "/recursion_0_model_best_dice.ckpt-26699")
##            
          
            recursion=0

            random_walked = np.array(recursion_data['random_walked'])

        
            recursion_data = predict_next_gt(data2=recursion_data,
                                                 images_train=images_train,
                                                 images_placeholder=images_placeholder,
                                                 training_time_placeholder=training_time_placeholder,
                                                 keep_prob=keep_prob,
                                                 logits=logits,
                                                 
                                                 sess=sess,
                                                 random_walked=random_walked,
                                                 )



    except Exception:
        raise
    # except (KeyboardInterrupt, SystemExit):
    #     try:
    #         recursion_data.close();
    #         logging.info('Keyboard interrupt / system exit caught - successfully closed data file.')
    #     except:
    #         logging.info('Keyboard interrupt / system exit caught - could not close data file.')



def apply_crf(seg_pred_probs,imgs):
    
    # Appearance parameters
#    a_sxy=160 # theta_alpha (refer website for equation terminology)
#    a_srgb= 3 # theta_beta
#    a_w1= 5 # weight term for bilateral term
#    # Gaussian smoothness term
#    g_sxy= 10 # theta_gamma
#    g_w2=30 # weight term for Gaussian smoothness
    a_w1, a_sxy, a_srgb = 5, 2, 0.1
    g_w2, g_sxy = 10, 5

    num_batch = seg_pred_probs.shape[0]
    masks_out = np.zeros_like(imgs)
    for k in range(num_batch):
        seg_pred_prob = seg_pred_probs[k,:,:,:]
        img = imgs[k,:,:,:]
        seg_pred_prob_tmp = np.swapaxes(np.swapaxes( seg_pred_prob,0,2 ),1,2)# to above mentioned shape - you could use np.swapaxes to achieve this.
    
        unary = unary_from_softmax(seg_pred_prob_tmp)
        
        d = dcrf.DenseCRF2D(exp_config.image_size[0],exp_config.image_size[1], exp_config.nlabels)
        d.setUnaryEnergy(unary)
        
        ###########################
        #Calculate Bilateral term
        ###########################
        
        ################################################
#        img_re = np.squeeze(img) #img_test_slice- 2D image containing intensity values
        img_re = np.copy(img)
        gaussian_pairwise_energy = create_pairwise_gaussian(sdims=(g_sxy,g_sxy), shape=img_re.shape[:2])
        d.addPairwiseEnergy(gaussian_pairwise_energy, compat=g_w2)
        
        bilateral_pairwise_energy = create_pairwise_bilateral(sdims=(a_sxy,a_sxy), schan=(a_srgb,), img=img_re, chdim=2)
        d.addPairwiseEnergy(bilateral_pairwise_energy, compat=a_w1) 

#        gaussian_pairwise_energy = create_pairwise_gaussian(shape=img_re.shape[:2])
#        d.addPairwiseEnergy(gaussian_pairwise_energy)
#        
#        bilateral_pairwise_energy = create_pairwise_bilateral(img=img_re, chdim=2)
#        d.addPairwiseEnergy(bilateral_pairwise_energy) 
        ################################################
        
        ######################
        # Inference 
        ######################
        # Run inference for 100 iterations
        Q_final = d.inference(100)
        
        # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
        crf_seg_soln = np.argmax(Q_final, axis=0)
        
        # Unfortunately, the DenseCRF flattens everything, so get it back into picture form (width,height).
        crf_seg_soln = crf_seg_soln.reshape((exp_config.image_size[0],exp_config.image_size[1]))
        masks_out[k,:,:,:] = np.copy(np.expand_dims(crf_seg_soln,2))
    return masks_out
def predict_next_gt(data2,
                    images_train,
                    images_placeholder,
                    training_time_placeholder,
                    keep_prob,
                    logits,
                    
                    sess,
                    random_walked):
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
    recursion = utils.get_recursion_from_hdf5(data2)

    new_recursion_fname = acdc_data.recursion_filepath(recursion + 1, data_file=data2)
    fpath = os.path.dirname(data2.filename)
    data = acdc_data.create_recursion_dataset(fpath, recursion + 1)

    #attributes to track processing
    prediction = data['predicted']
    logits_h5 = data['logits']
    pre_logits_h5 = data['pre_logits']
    predicted_pre_crf = data['predicted_pre_crf']
    processed = data['predicted'].attrs.get('processed')
    ran = data['random_walked']
    
    ran2 = data2['random_walked']
    means_h_2 = data2['means']
    p_vals = data2['p_vals']
    

    if not processed:
        processed_to = data['predicted'].attrs.get('processed_to')
        scr_max = len(images_train)
        print("SCR max = " + str(scr_max))
        for scr_idx in range(processed_to, scr_max, exp_config.batch_size):
            if scr_idx+exp_config.batch_size > scr_max:
                print("Entered last")
                # At the end of the dataset
                # Must ensure feed_dict is 20 images long however
                ind = list(range(scr_max - exp_config.batch_size, scr_max))
            else:
                
                ind = list(range(scr_idx, scr_idx + exp_config.batch_size))
                print(str(ind))
#            logging.info('Saving prediction after recursion {0} for images {1} to {2} '
#                         .format(ind[0], ind[-1]))
            x = np.expand_dims(np.array(images_train[ind, ...]), -1)

            
            temp_mean = np.array(means_h_2[ind, ...])
            logs = tf.nn.softmax(temp_mean)
            
            cur_rand = np.array(ran2[ind, ...])
            for t in range(exp_config.batch_size):
                for m in range(212):
                    for n in range(212):
                        if cur_rand[t,m,n] != 0:
                            logs[t,m,n,:] =0
                            logs[t,m,n,cur_rand[t,m,n]] = 1
                            
                    
            
            temp_maps = np.argmax(temp_mean, axis=-1)
            
            mask_out = apply_crf(logs,x)
            
            conf = 0.05
            temp_p = np.array(p_vals[ind, ...])
            for t in range(exp_config.batch_size):
                for m in range(212):
                    for n in range(212):
          

                        if temp_p[t,m,n] > conf:
                            mask_out[t,m,n] = 0
                            
                            
            for t in range(exp_config.batch_size):
                for m in range(212):
                    for n in range(212):
                        if cur_rand[t,m,n] != 0:
                            mask_out[t,m,n] = cur_rand[t,m,n]
            #save to dataset
            for indice in range(len(ind)):
                
                prediction[ind[indice], ...] = np.squeeze(mask_out[indice, ...])
                predicted_pre_crf[ind[indice], ...] = np.squeeze(temp_maps[indice, ...])
                logits_h5[ind[indice], ...] = np.squeeze(temp_mean[indice, ...])
                ran[ind[indice], ...] = np.squeeze(cur_rand[indice, ...])
             
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



def main():

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description="Train a neural network.")
    # parser.add_argument("CONFIG_PATH", type=str, help="Path to config file (assuming you are in the working directory)")
    # args = parser.parse_args()

    main()
