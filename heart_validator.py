# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)
import nibabel as nib
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
from experiments import unet2D_ws_spot_blur as exp_config
# from experiments import unet3D_bn_modified as exp_config
# from experiments import unet2D_bn_wxent as exp_config
# from experiments import FCN8_bn_wxent as exp_config

########################################################################################
validation_res_path = '/scratch_net/biwirender02/cany/scribble/logdir/'+exp_config.experiment_name+'/val_results'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


just_started = True


output_path = validation_res_path
path_pred = os.path.join(output_path, 'prediction')
path_gt = os.path.join(output_path, 'ground_truth')

path_image = os.path.join(output_path, 'image')

utils.makefolder(path_gt)
utils.makefolder(path_pred)

utils.makefolder(path_image)
    
    

step = 6600

try:
    import cv2
except:
    logging.warning('Could not find cv2. If you want to use augmentation '
                    'function you need to setup OpenCV.')




def run_training(continue_run):

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)
    already_created_recursion = False
    print("ALready created recursion : " + str(already_created_recursion))

    base_data = h5py.File(os.path.join(log_dir, 'base_data.hdf5'), 'r')
    
    try:
       
  
        images_val = np.array(base_data['images_test'])
        labels_val = np.array(base_data['masks_test'])
     

  

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



            # Add the variable initializer Op.
            init = tf.global_variables_initializer()



            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            saver = tf.train.Saver(max_to_keep=2)

            # Run the Op to initialize the variables.
            sess.run(init)
            


            saver.restore(sess,'/scratch_net/biwirender02/cany/scribble/logdir/unet2D_ws_spot_blur_crf/recursion_1_model_best_dice.ckpt-31099')




            


      
                  

            # Evaluate against the validation set.
            logging.info('Validation Data Eval:')
            eval_val_loss = model.evaluation(logits,
                                 labels_placeholder,
                                 images_placeholder,
                                 nlabels=exp_config.nlabels,
                                 loss_type=exp_config.loss_type,
                                 weak_supervision=True,
                                 cnn_threshold=exp_config.cnn_threshold,
                                 include_bg=False)
            
            [val_loss, val_dice] = do_eval(sess,
                                                       eval_val_loss,
                                                       images_placeholder,
                                                       labels_placeholder,
                                                       training_time_placeholder,
                                                       images_val,
                                                       labels_val,
                                                       exp_config.batch_size)
            logging.info('Found new best dice on validation set! - {} - '
                          .format(val_dice))
           
                
#            sio.savemat( validation_res_path+ '/result'+'_'+str(step)+'.mat', {'pred':np.float32(hard_pred),
#        'labels':np.float32(labels), 'dices':np.asarray(cdice)})
#           
    
            

  

    except Exception:
        raise
    # except (KeyboardInterrupt, SystemExit):
    #     try:
    #         recursion_data.close();
    #         logging.info('Keyboard interrupt / system exit caught - successfully closed data file.')
    #     except:
    #         logging.info('Keyboard interrupt / system exit caught - could not close data file.')





def save_nii(img_path, data):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data,affine=None)
    nimg.to_filename(img_path)

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
        dice_ii += np.mean(cdice,0)
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: '  +str(avg_loss)+" , average dice: " + str(avg_dice))

    return avg_loss, np.mean(avg_dice)


def my_dice(ar1, ar2):
    
    interse = np.multiply(ar1,ar2)
    
    return 2*np.sum(interse)/(np.sum(ar1) + np.sum(ar2))
    
    

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
