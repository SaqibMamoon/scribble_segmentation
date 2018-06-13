   # Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import argparse
import metrics_acdc_simple_pro
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform
from acdc_data import most_recent_recursion
import config.system as sys_config
import model as model
import utils
import image_utils
import h5py
from tfwrapper import losses
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
#sys_config.setup_GPU_environment()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            images,
            labels,
            batch_size):



    feed_dict = { images_placeholder: images,
                  labels_placeholder: labels,
            }

    closs, cdice = sess.run(eval_loss, feed_dict=feed_dict)
    
    return closs,cdice

def per_structure_dice(hard_pred, labels, epsilon=1e-10):
    '''
    Dice coefficient per subject per label
    :param logits: network output
    :param labels: groundtruth labels (one-hot)
    :param epsilon: for numerical stability
    :return: tensor shaped (tf.shape(logits)[0], tf.shape(logits)[-1])
    '''

    
    intersection = tf.multiply(hard_pred, labels)


    reduction_axes = [0,1]

    intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=reduction_axes)  # was [1,2]

    l = tf.reduce_sum(hard_pred, axis=reduction_axes)
    r = tf.reduce_sum(labels, axis=reduction_axes)

    dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

    return dices_per_subj


def get_dice(pred,labels,nlabels):
    cdice_structures = per_structure_dice(pred, tf.one_hot(labels, depth=nlabels))

    cdice_foreground = tf.slice(cdice_structures, [1], [ nlabels - 2])


    cdice = tf.reduce_mean(cdice_foreground)
    return cdice
def save_nii(img_path, data):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data,affine=None)
    nimg.to_filename(img_path)
def read_data(path):


  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('images_test'))
    label = np.array(hf.get('masks_test'))
    return data, label


def score_data(input_folder, output_folder, model_path, exp_config, do_postprocessing=False, recursion=None):

    print("KOD YENİ")
    dices = []
    images, labels = read_data('/scratch/cany/scribble/scribble_data/prostate_divided.h5')
    num_images = images.shape[0]
    print(str(num_images))
    print(str(images.shape))
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl,logits = model.predict_logits(images_pl, exp_config.model_handle, exp_config.nlabels)
    
    
    mask_tensor_shape = [batch_size] + list(exp_config.image_size)
  
#    images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
    labels_placeholder = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')


    # Add to the Graph the Ops for loss calculation.

#    eval_val_loss = model.evaluation(logits,
#                                     labels_placeholder,
#                                     images_pl,
#                                     nlabels=exp_config.nlabels,
#                                     loss_type=exp_config.loss_type,
#                                     weak_supervision=True,
#                                     cnn_threshold=exp_config.cnn_threshold,
#                                     include_bg=False)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        if recursion is None:
            checkpoint_path = utils.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')
        else:
            try:
                checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,
                                                                         'recursion_{}_model_best_dice.ckpt'.format(recursion))
            except:
                checkpoint_path = utils.get_latest_model_checkpoint_path(model_path,
                                                                         'recursion_{}_model.ckpt'.format(recursion))

        saver.restore(sess, checkpoint_path)

        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])

        for k in range(num_images):
            network_input = np.expand_dims(np.expand_dims(images[k,:,:],2),0)
            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
            prediction_cropped = np.squeeze(logits_out[0, ...])
            
            # ASSEMBLE BACK THE SLICES
            prediction_arr = np.uint8(np.argmax(prediction_cropped, axis=-1))
            


#            prediction_arr = np.squeeze(np.transpose(np.asarray(prediction, dtype=np.uint8), (1,2,0)))
#    
                           
            mask = labels[k,:,:]
    
                            # This is the same for 2D and 3D again
            if do_postprocessing:
                print("Entered post processing " + str(True))
                prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)
    
            
            # Save predicted mask
            out_file_name = os.path.join(output_folder, 'prediction', 'patient' + str(k) +'.nii.gz')

            logging.info('saving to: %s' % out_file_name)
            save_nii(out_file_name, prediction_arr)
    
            # Save GT image
            gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + str(k) + '.nii.gz')
            logging.info('saving to: %s' % gt_file_name)
            save_nii(gt_file_name, np.uint8(mask))
    
#            # Save difference mask between predictions and ground truth
#            difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
#            difference_mask = np.asarray(difference_mask, dtype=np.uint8)
#            diff_file_name = os.path.join(output_folder,
#                                          'difference',
#                                          'patient' + str(k) + '.nii.gz')
#            logging.info('saving to: %s' % diff_file_name)
#            save_nii(diff_file_name, difference_mask)
    
            # Save image data to the same folder for convenience
            image_file_name = os.path.join(output_folder, 'image',
                                    'patient' + str(k)  + '.nii.gz')
            logging.info('saving to: %s' % image_file_name)
            save_nii(image_file_name, images[k,:,:])

#            feed_dict = { images_pl: network_input,
#                          labels_placeholder: np.expand_dims(np.squeeze(labels[k,:,:]),0),
#                    }
#
#            closs, cdice = sess.run(eval_val_loss, feed_dict=feed_dict)

#            print(str(prediction_arr.shape))
#            tempp= np.expand_dims(np.squeeze(labels[k,:,:]),0)
#            print(str(tempp.shape))
#            qwe=tf.one_hot(np.uint8(np.squeeze(labels[k,:,:])), depth=4)
#            print(str(sess.run(tf.shape(qwe))))
#            tempp2 = tf.one_hot(prediction_arr, depth=4)
#            print(str(sess.run(tf.shape(tempp2))))
            cdice = sess.run(get_dice(tf.one_hot(np.uint8(prediction_arr), depth=4),np.uint8(np.squeeze(labels[k,:,:])),4))
            print(str(cdice))
#            [val_loss, val_dice] = do_eval(sess,
#                                                       eval_val_loss,
#                                                       images_placeholder,
#                                                       labels_placeholder,
#                                                     network_input,
#                                                       np.expand_dims(np.squeeze(labels[k,:,:]),0),
#                                                       exp_config.batch_size)
            dices.append(cdice)
    print("Average Dice : " + str(np.mean(dices)))
    return init_iteration


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script to evaluate a neural network model on the ACDC challenge data")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    parser.add_argument("SAVE_IMAGES",
                        help="OPTIONAL: Set print directory to print best/median/worst cases",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--save", dest="SAVE_IMAGES", default=0)
    parser.add_argument("RECURSION",
                        help="OPTIONAL: For weak supervision, specify recursion to be evaluated. "
                             "By default, will do most recent.",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--recursion", dest="RECURSION", default=None)

    parser.add_argument("POSTPROCESS",
                        help="OPTIONAL: Set to 1 to postprocess",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--postprocess", dest="POSTPROCESS", default=0)

    args = parser.parse_args()

    postprocess = (args.POSTPROCESS == 1)

    model_path = '/scratch_net/biwirender02/cany/scribble/logdir/prostate_full_rw_90'
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    input_path = sys_config.data_root
    recursion = args.RECURSION
    if recursion is None:
        recursion = most_recent_recursion(model_path)
        print("Recursion {} from folder {}".format(recursion, model_path))
        if recursion == -1:
            output_path = os.path.join(model_path, 'predictions')
            recursion = None
        else:
            output_path = os.path.join(model_path, 'predictions_recursion_{}'.format(recursion))

    else:
        output_path = os.path.join(model_path, 'predictions_recursion_{}'.format(args.RECURSION))

    printing = args.SAVE_IMAGES == 1
    path_pred = os.path.join(output_path, 'prediction')
    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_image = os.path.join(output_path, 'image')
    path_eval = os.path.join(output_path, 'eval')

    utils.makefolder(path_gt)
    utils.makefolder(path_pred)
    utils.makefolder(path_diff)
    utils.makefolder(path_image)
    init_iteration = score_data(input_path, output_path, model_path, do_postprocessing=True, exp_config=exp_config, recursion=recursion)






