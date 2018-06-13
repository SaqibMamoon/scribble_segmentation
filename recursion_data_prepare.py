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
import model as model
from background_generator import BackgroundGenerator
import config.system as sys_config
import acdc_data
import random_walker
import h5py
from scipy.ndimage.filters import gaussian_filter

### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:

#from experiments import test as exp_config
from experiments import unet2D_ws_spot_blur as exp_config

# from experiments import unet3D_bn_modified as exp_config
# from experiments import unet2D_bn_wxent as exp_config
# from experiments import FCN8_bn_wxent as exp_config

########################################################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')


def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)

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
            recursion_data = acdc_data.load_different_recursion(recursion_data, -1)
            recursion-=1

        # load images and validation data
        images_train = np.array(base_data['images_train'])
        scribbles_train = np.array(base_data['scribbles_train'])
        images_val = np.array(base_data['images_test'])
        labels_val = np.array(base_data['masks_test'])


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

            tf.summary.scalar('learning_rate', learning_rate_placeholder)

            # Build a Graph that computes predictions from the inference model.
            logits = model.inference(images_placeholder,
                                     exp_config.model_handle,
                                     training=training_time_placeholder,
                                     nlabels=exp_config.nlabels)


            summary = tf.summary.merge_all()

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()



            # Run the Op to initialize the variables.
            sess.run(init)
            
            saver = tf.train.Saver(max_to_keep=2)

            if continue_run:
                # Restore session
                saver.restore(sess, init_checkpoint_path)

            step = init_step
            curr_lr = exp_config.learning_rate

            no_improvement_counter = 0
            best_val = np.inf
            last_train = np.inf
            loss_history = []
            loss_gradient = np.inf
            best_dice = 0
            logging.info('RECURSION {0}'.format(recursion))

#            # random walk - if it already has been random walked it won't redo
#            recursion_data = acdc_data.random_walk_epoch(recursion_data, exp_config.rw_beta, exp_config.rw_threshold, exp_config.random_walk)


                    #Have reached end of recursion
            recursion_data = predict_next_gt(data=recursion_data,
                                             images_train=images_train,
                                             images_placeholder=images_placeholder,
                                             training_time_placeholder=training_time_placeholder,
                                             logits=logits,
                                             sess=sess)

            recursion_data = postprocess_gt(data=recursion_data,
                                            images_train=images_train,
                                            scribbles_train=scribbles_train)
            recursion += 1
            # random walk - if it already has been random walked it won't redo
            recursion_data = acdc_data.random_walk_epoch(recursion_data,
                                                         exp_config.rw_beta,
                                                         exp_config.rw_threshold,
                                                         exp_config.random_walk)
    except Exception:
        raise

def predict_next_gt(data,
                    images_train,
                    images_placeholder,
                    training_time_placeholder,
                    logits,
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
    processed = data['predicted'].attrs.get('processed')
    if not processed:
        processed_to = data['predicted'].attrs.get('processed_to')
        data.close()
        scr_max = len(images_train)
        print("SCR max = " + str(scr_max))
        print(str(processed_to))
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

            feed_dict = {
                images_placeholder: x,
                training_time_placeholder: False
            }
            softmax = tf.nn.softmax(logits)

            print("softmax")
            #threshold output of cnn
            if exp_config.cnn_threshold:
                threshold = tf.constant(exp_config.cnn_threshold, dtype=tf.float32)
                s = tf.multiply(tf.ones(shape=[exp_config.batch_size, 212, 212, 1]), threshold)
                softmax = tf.concat([s, softmax[..., 1:]], axis=-1)
            print("threshold")
            # if exp_config.use_crf:
            #     #get unary from softmax
            #     unary = tf.multiply(-1, tf.log(softmax))
            #
            #calculate mask
            mask = tf.arg_max(softmax, dimension=-1)
            
            print("before sess")
            mask_out = sess.run(mask, feed_dict=feed_dict)
            print("after sess : " + str(mask_out))
            #save to dataset
            data = h5py.File(new_recursion_fname, 'r+')
            prediction = data['predicted']
            for indice in range(len(ind)):
                
                prediction[ind[indice], ...] = np.squeeze(mask_out[indice, ...])
                print("added " + str(ind[indice]))
            data['predicted'].attrs.modify('processed_to', scr_idx + exp_config.batch_size)
            data.close()
        if exp_config.reinit:
            logging.info("Initialising variables")
            sess.run(tf.global_variables_initializer())
        data = h5py.File(new_recursion_fname, 'r+')
        prediction = data['predicted']
        data['predicted'].attrs.modify('processed', True)
        logging.info('Created unprocessed ground truths for recursion {}'.format(recursion + 1))

        #Reopen in read only mode
        data.close()
        data = h5py.File(new_recursion_fname, 'r')
    return data

def postprocess_gt(data, images_train, scribbles_train=None):
    '''
    Postprocesses predictions of CNN to create ground truths for recursion
    :param data: Data of this recursion - e.g. if given data file for recursion n,
                 It will set up ground truths to be random walked for recursion n
    :param images_train: Numpy array of training images
    :param scribbles_train: Numpy array of weakly annotated images
    :return: data file with postprocessed ground truths
    '''

    #reopen in write mode
    data_fpath = data.filename
    data.close()
    data = h5py.File(data_fpath)

    #get recursions and previous progress
    recursion = utils.get_recursion_from_hdf5(data)
    predictions = np.array(data['predicted'])
    postprocessed = data['postprocessed']
    done = postprocessed.attrs.get('processed')
    if not done:
        processed_to = data['postprocessed'].attrs.get('processed_to')
        scr_max = len(images_train)

        #LOGGING: Print the postprocessing options
        if exp_config.postprocessing:
            logging.info("Postprocessing options chosen:")
            if exp_config.reinit: logging.info("\treinit = True\n\t\t\t\t\t"
                                               "Reinitialising network weights on each recursion")
            if exp_config.keep_largest_cluster: logging.info("\tkeep_largest_blob = True\n\t\t\t\t\t"
                                                             "Keeping only largest cluster from ground truth prediction")
            if exp_config.cnn_threshold: logging.info("\tcnn_threshold = {0}\n\t\t\t\t\t"
                                                      "Segmentation predictions of CNN with probability"
                                                      " of less than {0} are left unlabelled".format(exp_config.cnn_threshold))
            if exp_config.rw_intersection: logging.info("\trw_intersection = True\n\t\t\t\t\t"
                                                        "Random walker segmentation used as "
                                                        "upper bound")
            if exp_config.rw_reversion: logging.info("\trw_reversion = True\n\t\t\t\t\tIf after post processing the"
                                                     " network has predicted less than the original ground truth, "
                                                     "it will revert to the original ground truth to prevent "
                                                     "propagation of bad predictions")

            if exp_config.smooth_edges: logging.info("\t smooth_edges with sigma = {} and threshold = {}\n\t\t\t\t\t"
                                                     "Will use a gaussian filter to smooth edges of the segmentation")
        else:
            logging.info("No postprocessing options enabled")


        for scr_idx in range(processed_to, scr_max, exp_config.batch_size):
            # get indices to process
            if scr_idx+exp_config.batch_size > scr_max:
                ind = list(range(scr_max - exp_config.batch_size, scr_max))
            else:
                ind = list(range(scr_idx, scr_idx + exp_config.batch_size))

            logging.info('Postprocessing groundtruth of recursion {0} for images {1} to {2} '
                         .format(recursion, ind[0], ind[-1]))

            mask_out = postprocess(np.array(predictions[ind, ...]), images_train[ind, ...], scribbles_train[ind, ...])

            #save to dataset
            postprocessed[ind, ...] = np.squeeze(mask_out)
            data['postprocessed'].attrs.modify('processed_to', scr_idx + exp_config.batch_size)
        data['postprocessed'].attrs.modify('processed', True)
        logging.info('Finished postprocessing ground truths for recursion {}'.format(recursion))
    #reopen in read only mode
    data.close()
    data = h5py.File(data_fpath, 'r')
    return data
def postprocess(mask_out, images_train, scribbles_train=None):
    '''
    Postprocesses predictions of CNN to create ground truths for recursion
    :param data: Data of this recursion - e.g. if given data file for recursion n,
                 It will set up ground truths to be random walked for recursion n
    :param images_train: Numpy array of training images
    :param scribbles_train: Numpy array of weakly annotated images
    :return:
'''

    #get labels present
    labels = np.unique(scribbles_train)
    labels = labels[labels != 0]

    # use full segmentation of random walker as upper bound
    if exp_config.rw_intersection:
        rw_segmentation = random_walker.segment(images_train,
                                               scribbles_train,
                                               threshold=0,
                                               beta=exp_config.rw_beta)
        mask = mask_out[:]
        mask_out = np.zeros_like(mask_out)
        for label in labels:
            indices = (rw_segmentation == label)
            indices &= (mask == label)
            mask_out[indices] = label

    #revert to original random walked data for 'bad' prediction
    if exp_config.rw_reversion:
        mask = mask_out[:]
        mask_out = np.zeros_like(mask_out)
        for img_id in range(exp_config.batch_size):
            for label in labels:
                if np.sum(mask[img_id, ...] == label) < np.sum(scribbles_train[img_id, ...] == label):
                    #If the prediction has predicted less than the original scribble, revert to
                    #the scribble
                    mask_out[img_id, scribbles_train[img_id, ...] == label] = label
                else:
                    mask_out[img_id, mask[img_id, ...] == label] = label


    #keep only largest cluster for output
    if exp_config.keep_largest_cluster:
        for img_id in range(exp_config.batch_size):
            mask_out[img_id, ...] = image_utils.keep_largest_connected_components(np.squeeze(mask_out[img_id, ...]))

    if exp_config.smooth_edges:
        labels = labels[labels != np.max(labels)]
        for img_id in range(exp_config.batch_size):
            mask = mask_out[img_id, ...]
            new_mask = np.zeros_like(mask)
            for label in labels:
                struct = (mask == label).astype(np.float)
                blurred_struct = gaussian_filter(struct, sigma=exp_config.edge_smoother_sigma)
                # ax = fig.add_subplot(161 + label)
                blurred_struct[blurred_struct >= exp_config.edge_smoother_threshold] = 1
                blurred_struct[blurred_struct <exp_config.edge_smoother_threshold] = 0
                new_mask[blurred_struct != 0] = label
            mask_out[img_id, ...] = new_mask

    return mask_out



def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

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

    for batch in iterate_minibatches(images, labels, batch_size=batch_size, augment_batch=False):  # No aug in evaluation
    # As before you can wrap the iterate_minibatches function in the BackgroundGenerator class for speed improvements
    # but at the risk of not catching exceptions

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      labels_placeholder: y,
                      training_time_placeholder: False}

        closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
        loss_ii += closs
        dice_ii += cdice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: %0.04f, average dice: %0.04f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice


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
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # RANDOM CROP SCALE
        if do_scaleaug:
            offset = kwargs.get('offset', 30)
            n_x, n_y = img.shape
            r_y = np.random.random_integers(n_y-offset, n_y)
            p_x = np.random.random_integers(0, n_x-r_y)
            p_y = np.random.random_integers(0, n_y-r_y)

            img = image_utils.resize_image(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y))
            lbl = image_utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

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
