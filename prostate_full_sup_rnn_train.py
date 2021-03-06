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
import model_full as model
from background_generator import BackgroundGenerator
import config.prostate_system as sys_config
import acdc_data
import random_walker
import h5py
from scipy.ndimage.filters import gaussian_filter
import scipy.io as sio
import scipy
from medpy.metric.binary import hd, dc, assd
### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:

#from experiments import test as exp_config
#from experiments import unet2D_ws_spot_blur as exp_config
from experiments import prostate_full_sup_rnn_exp as exp_config
# from experiments import unet3D_bn_modified as exp_config
# from experiments import unet2D_bn_wxent as exp_config
# from experiments import FCN8_bn_wxent as exp_config

########################################################################################
validation_res_path = '/scratch_net/biwirender02/cany/scribble/logdir/'+exp_config.experiment_name+'/val_results'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
#sys_config.setup_GPU_environment()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
                                                                                      target_folder=log_dir,
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
#        scribbles_train = np.array(base_data['scribbles_train'])
        labels_train = np.array(base_data['masks_train'])
        images_val = np.array(base_data['images_val'])
        labels_val = np.array(base_data['masks_val'])
        slices_val = np.array(base_data['slices_validation'])
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
        
        
        
        if not os.path.exists(validation_res_path):
            os.makedirs(validation_res_path)
#        with tf.Graph().as_default():
        with tf.Session(config = config) as sess:
            # Generate placeholders for the images and labels.

            image_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size) + [1]
            mask_tensor_shape = [exp_config.batch_size] + list(exp_config.image_size)
            print("Exp config image size : " + str(exp_config.image_size))
            images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
            labels_placeholder = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')

            learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
            crf_learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
            training_time_placeholder = tf.placeholder(tf.bool, shape=[])

            tf.summary.scalar('learning_rate', learning_rate_placeholder)

            # Build a Graph that computes predictions from the inference model.
            logits = model.inference(images_placeholder,
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

            
   
            crf_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='crf_scope')
            

            restore_var = [v for v in tf.all_variables() if v.name not in crf_variables]
            
   
            global_step = tf.Variable(0, name='global_step', trainable=False)

            network_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(loss,
                                                                                         var_list=restore_var,
                                                                                         colocate_gradients_with_ops=True,
                                                                                         global_step=global_step)

            crf_train_op = tf.train.AdamOptimizer(learning_rate=crf_learning_rate_placeholder).minimize(loss,var_list=crf_variables,
                                                                                         colocate_gradients_with_ops=True,
                                                                                         global_step=global_step)
            

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
            
#            crf_training_variables =tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='crf_training_op')
#            print(str(crf_training_variables))
#            all_crf_variables = crf_training_variables+crf_variables
#            print(str(all_crf_variables))
            
  

#/////////////////////

#            load_saver = tf.train.Saver(var_list=load_variables)
#            load_saver.restore(sess, '/scratch_net/biwirender02/cany/basil/logdir/unet2D_ws_spot_blur/recursion_0_model.ckpt-7499')

#///////////////////////////

#            if continue_run:
#                # Restore session
#                saver.restore(sess, init_checkpoint_path)
#            saver.restore(sess,'/scratch_net/biwirender02/cany/scribble/logdir/prostate_deep_rnn_exp/recursion_0_model_best_dice.ckpt-4999')
##            
#            saver.restore(sess,'/scratch_net/biwirender02/cany/scribble/logdir/prostate_deep_rnn_exp/recursion_0_model_best_dice.ckpt-5499')
##            
            init_step = 0
            start_epoch = 0
            recursion=0
            
            step = init_step
            curr_lr = exp_config.learning_rate/10
            crf_curr_lr = 1e-08
            no_improvement_counter = 0
            best_val = np.inf
            last_train = np.inf
            loss_history = []
            loss_gradient = np.inf
            best_dice = 0
            logging.info('RECURSION {0}'.format(recursion))

            


            # random walk - if it already has been random walked it won't redo
 
            print("Start epoch : " +str(start_epoch) + " : max epochs : " + str(exp_config.epochs_per_recursion))
            for epoch in range(start_epoch, exp_config.max_epochs):
               
                logging.info('Epoch {0} ({1} of {2} epochs for recursion {3})'.format(epoch,
                                                                                      1 + epoch % exp_config.epochs_per_recursion,
                                                                                      exp_config.epochs_per_recursion,
                                                                                      recursion))
                # for batch in iterate_minibatches(images_train,
                #                                  labels_train,
                #                                  batch_size=exp_config.batch_size,
                #                                  augment_batch=exp_config.augment_batch):

                # You can run this loop with the BACKGROUND GENERATOR, which will lead to some improvements in the
                # training speed. However, be aware that currently an exception inside this loop may not be caught.
                # The batch generator may just continue running silently without warning even though the code has
                # crashed.

                for batch in BackgroundGenerator(iterate_minibatches(images_train,
                                                                     labels_train,
                                                                     batch_size=exp_config.batch_size,
                                                                     augment_batch=exp_config.augment_batch)):

                    if exp_config.warmup_training:
                        if step < 50:
                            curr_lr = exp_config.learning_rate / 10.0
                        elif step == 50:
                            curr_lr = exp_config.learning_rate
                    if ((step % 5000 == 0) & (step > 0)):
                        curr_lr = curr_lr*0.94
                        crf_curr_lr = crf_curr_lr*0.94
                    start_time = time.time()

                    # batch = bgn_train.retrieve()
                    x, y = batch

                    # TEMPORARY HACK (to avoid incomplete batches
                    if y.shape[0] < exp_config.batch_size:
                        step += 1
                        continue

                    network_feed_dict = {
                        images_placeholder: x,
                        labels_placeholder: y,
                        learning_rate_placeholder: curr_lr,
                        training_time_placeholder: True
                    }

                    crf_feed_dict = {
                        images_placeholder: x,
                        labels_placeholder: y,
                        crf_learning_rate_placeholder: crf_curr_lr,
                        training_time_placeholder: True
                    }                    

                    if (step % 10 == 0) :
                        _, loss_value = sess.run([crf_train_op, loss], feed_dict=crf_feed_dict)
                    _, loss_value = sess.run([network_train_op, loss], feed_dict=network_feed_dict)
                    duration = time.time() - start_time

                    # Write the summaries and print an overview fairly often.
                    if step % 10 == 0:
                        # Print status to stdout.
                        logging.info('Step %d: loss = %.6f (%.3f sec)' % (step, loss_value, duration))
                        # Update the events file.



                    # Save a checkpoint and evaluate the model periodically.
                    if (step + 1) % exp_config.val_eval_frequency == 0:

                        checkpoint_file = os.path.join(log_dir, 'recursion_{}_model.ckpt'.format(recursion))
                        saver.save(sess, checkpoint_file, global_step=step)
                        # Evaluate against the training set.


                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')
                        [val_loss, val_dice, hard_pred, labels, cdice] = do_eval(sess,
                                                       logits,
                                                       images_placeholder,
                                                       labels_placeholder,
                                                       training_time_placeholder,
                                                       images_val,
                                                       labels_val,
                                                       exp_config.batch_size,slices_val)

                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss, val_dice_: val_dice}
                        )
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            best_file = os.path.join(log_dir, 'recursion_{}_model_best_dice.ckpt'.format(recursion))
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - {} - '
                                         'Saving recursion_{}_model_best_dice.ckpt' .format(val_dice, recursion))
                            text_file = open('val_results.txt', "a")
                            text_file.write("\nVal dice " + str(step) +" : " + str(val_dice))
                            text_file.close()
                            
#                            sio.savemat( validation_res_path+ '/result'+'_'+str(step)+'.mat', {'pred':np.float32(hard_pred),
#                    'labels':np.float32(labels), 'dices':np.asarray(cdice)})
                        if val_loss < best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'recursion_{}_model_best_xent.ckpt'.format(recursion))
                            saver_best_xent.save(sess, best_file, global_step=step)
                            logging.info('Found new best crossentropy on validation set! - {} - '
                                         'Saving recursion_{}_model_best_xent.ckpt'.format(val_loss, recursion))


                    step += 1

    except Exception:
        raise
    # except (KeyboardInterrupt, SystemExit):
    #     try:
    #         recursion_data.close();
    #         logging.info('Keyboard interrupt / system exit caught - successfully closed data file.')
    #     except:
    #         logging.info('Keyboard interrupt / system exit caught - could not close data file.')



def do_eval(sess,
            logits_op,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size,
            slices_val):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''

    loss_ii = 0
    dice_ii = 0
    num_batches = 0
    count  = 0
#    all_pred = []
#    all_labels=[]
    dices = []
    logging.info(str(slices_val))
    logging.info(str(type(slices_val)))
    for patient in range(len(slices_val)):
        cur_slice=np.copy(np.uint(sum(slices_val[:patient])))
        
        
        whole_label=np.copy(labels[np.uint(cur_slice):np.uint(cur_slice+slices_val[patient]),:,:])
        res = np.zeros((np.uint(slices_val[patient]),320,320))
        for sli in range(int(slices_val[patient])):
            x=np.zeros((4,320,320,1))
            x[0,:,:,:]=np.copy(np.expand_dims(images[int(sli+cur_slice),:,:],axis=3))
#            y=labels[int(sli+cur_slice),:,:]
            

            softmax = tf.nn.softmax(tf.slice(logits_op,[0,0,0,0],[1,-1,-1,-1]), dim=-1)
            mask_op = tf.arg_max(softmax, dimension=-1)  # was 3

            

            
            feed_dict = { images_placeholder: x,
              
              training_time_placeholder: False}
    
            mask = sess.run(mask_op, feed_dict=feed_dict)

            res[sli,:,:] = np.copy(mask)
                    
        temp_dices = []
        temp_preds = []
        temp_labels=[]
        for c in [1,2]:
        # Copy the gt image to not alterate the input
            gt_c_i = np.copy(whole_label)
            gt_c_i[gt_c_i != c] = 0
    
            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(res)
            pred_c_i[pred_c_i != c] = 0
    
            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)
    
            temp_preds.append(pred_c_i)
            temp_labels.append(gt_c_i)
            # Compute the Dice
            
            dice = my_dice(gt_c_i, pred_c_i)
#            dice = dc(gt_c_i, pred_c_i)
            temp_dices.append(dice)
        
        dices.append(temp_dices)
        logging.info("Dice for patient : " + str(temp_dices[0]) + " and " +str(temp_dices[1]))
        if patient == 0:
            all_preds = np.asarray(temp_preds)
            all_labels = np.asarray(temp_labels)
        else:
            all_preds = np.concatenate([all_preds, np.asarray(temp_preds)],axis=1)
            all_labels = np.concatenate([all_labels, np.asarray(temp_labels)],axis=1)
        
        
#        all_pred.append(temp_preds)
#        all_labels.append(temp_labels)
#        


    avg_loss = 0.0000000
    avg_dice = np.mean(dices)

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice, np.asarray(all_preds), np.asarray(all_labels), dices


def my_dice(ar1, ar2):
    
    interse = np.multiply(ar1,ar2)
    
    return 2*np.sum(interse)/(np.sum(ar1) + np.sum(ar2))
    
    
def augmentation_function(images, labels, **kwargs):
    '''
    Function for augmentation of minibatches. It will transform a set of images and corresponding labels
    by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
    with random parameters. 
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size. 
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction. 
    :return: A mini batch of the same size but with transformed images and masks. 
    '''


    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    do_rotations = kwargs.get('do_rotations', False)
    do_scaleaug = kwargs.get('do_scaleaug', False)
    do_fliplr = kwargs.get('do_fliplr', False)


    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for ii in range(num_images):

        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        # ROTATE
        if do_rotations:
            angles = kwargs.get('angles', (-15,15))
            random_angle = np.random.uniform(angles[0], angles[1])
            
            
            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM CROP SCALE
        if do_scaleaug:
            offset = kwargs.get('offset', 30)
            n_x, n_y = img.shape
            r_y = np.random.random_integers(n_y-offset, n_y)
            p_x = np.random.random_integers(0, n_x-r_y)
            p_y = np.random.random_integers(0, n_y-r_y)

            img = scipy.misc.imresize(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y),interp='bilinear')
            lbl = scipy.misc.imresize(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y),interp='nearest')
            
        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                lbl = np.fliplr(lbl)


        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch


def iterate_minibatches(images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)

        if augment_batch:
            X, y = augmentation_function(X, y,
                                         do_rotations=exp_config.do_rotations,
                                         do_scaleaug=exp_config.do_scaleaug,
                                         do_fliplr=exp_config.do_fliplr)

        yield X, y


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
