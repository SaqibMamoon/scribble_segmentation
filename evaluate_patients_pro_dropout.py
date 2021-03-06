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
import model_dropout as model
import utils
import image_utils
import h5py
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def do_crf(seg_pred_probs,imgs):
    
    # Appearance parameters
#    a_sxy=160 # theta_alpha (refer website for equation terminology)
#    a_srgb= 3 # theta_beta
#    a_w1= 5 # weight term for bilateral term
#    # Gaussian smoothness term
#    g_sxy= 10 # theta_gamma
#    g_w2=30 # weight term for Gaussian smoothness

#    parameters= [6, 3, 0.01, 10, 2]
# 
#    a_w1, a_sxy, a_srgb = parameters[0],parameters[1],parameters[2]
#    g_w2, g_sxy = parameters[3],parameters[4]

#    a_sxy=0.5 # theta_alpha (refer website for equation terminology)
#    a_srgb= 1 # theta_beta
#    a_w1= 5 # weight term for bilateral term
#    # Gaussian smoothness term
#    g_sxy= 5 # theta_gamma
#    g_w2=30 # weight term for Gaussian smoothness
    parameters= [6, 3, 0.01, 10, 2]
# 
    a_w1, a_sxy, a_srgb = parameters[0],parameters[1],parameters[2]
    g_w2, g_sxy = parameters[3],parameters[4]

#    a_sxy=0.5 # theta_alpha (refer website for equation terminology)
#    a_srgb= 1 # theta_beta
#    a_w1= 5 # weight term for bilateral term
#    # Gaussian smoothness term
#    g_sxy= 5 # theta_gamma
#    g_w2=30 # weight term for Gaussian smoothness

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


def average_outputs(sess,softmax_pl,images_pl,network_input,keep_prob):

    temp = np.zeros((50,1,320,320,4))
    for k in range(50):
        

        temp[k,:,:,:,:] = sess.run(softmax_pl,feed_dict={images_pl: network_input,keep_prob:0.5})

    
    
    
    temp_mean = (np.mean(temp,axis=0))
    return temp_mean
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

def score_data(input_folder, output_folder, model_path, exp_config, do_postprocessing=False, recursion=None,apply_crf=False):

    base_data = h5py.File('/scratch_net/biwirender02/cany/scribble/scribble_data/prostate_divided_normalized.h5', 'r')
    
    
    images = np.array(base_data['images_test'])
    labels= np.array(base_data['masks_test'])
    slices_val = np.array(base_data['slices_test'])
    keep_prob = tf.placeholder(tf.float32, shape=[])
    training_time_placeholder = tf.placeholder(tf.bool, shape=[])
    batch_size = 4
    image_tensor_shape = [1] + list(exp_config.image_size) + [1]

    images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    _,logits_op = model.predict(images_placeholder,
                                keep_prob,
                                     exp_config.model_handle,
                                     
                                     nlabels=exp_config.nlabels)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()



    dices = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config = config) as sess:
        
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
        for patient in range(len(slices_val)):
            cur_slice=np.copy(np.uint(sum(slices_val[:patient])))
            
            
            whole_label=np.copy(labels[np.uint(cur_slice):np.uint(cur_slice+slices_val[patient]),:,:])
            res = np.zeros((np.uint(slices_val[patient]),320,320))
            for sli in range(int(slices_val[patient])):
                x=np.zeros((1,320,320,1))
                x[0,:,:,:]=np.copy(np.expand_dims(images[int(sli+cur_slice),:,:],axis=3))
    #            y=labels[int(sli+cur_slice),:,:]
                
#                logits_out = average_outputs(sess,logits_op,images_placeholder,x,keep_prob)
                logits_out=sess.run(logits_op,feed_dict={images_placeholder: x,keep_prob:1})
#                                
                if apply_crf:
                    softmax = sess.run(tf.nn.softmax(logits_out))
                    res[sli,:,:] = np.squeeze(do_crf(np.copy(softmax),x))
                else:
                    mask= np.argmax(logits_out, axis=-1)
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
                            # This is the same for 2D and 3D again
#            if do_postprocessing:
#                prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)
#    
            
            # Save predicted mask
            out_file_name = os.path.join(output_folder, 'prediction', 'patient' + str(patient) +'.nii.gz')
#            print(str(res.shape))
#            print(str(whole_label.shape))
#           
#            logging.info('saving to: %s' % out_file_name)
            
            save_nii(out_file_name, np.asarray(res))
    
            # Save GT image
            gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + str(patient) + '.nii.gz')
#            logging.info('saving to: %s' % gt_file_name)
            save_nii(gt_file_name, np.uint8(np.asarray(whole_label)))
    
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
                                    'patient' + str(patient)  + '.nii.gz')
#            logging.info('saving to: %s' % image_file_name)
            save_nii(image_file_name, np.uint8(255*np.asarray(images[np.uint(cur_slice):np.uint(cur_slice+slices_val[patient]),:,:])))



    return 0

def my_dice(ar1, ar2):
    
    interse = np.multiply(ar1,ar2)
    
    return 2*np.sum(interse)/(np.sum(ar1) + np.sum(ar2))
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
    
    parser.add_argument("apply_crf",
                        help="OPTIONAL: For weak supervision, specify recursion to be evaluated. "
                             "By default, will do most recent.",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--crf", dest="apply_crf", default=None)
    parser.add_argument("--recursion", dest="RECURSION", default=None)

    parser.add_argument("POSTPROCESS",
                        help="OPTIONAL: Set to 1 to postprocess",
                        nargs='?',
                        default=argparse.SUPPRESS)
    parser.add_argument("--postprocess", dest="POSTPROCESS", default=0)

    args = parser.parse_args()

    postprocess = (args.POSTPROCESS == 1)

    model_path = '/scratch_net/biwirender02/cany/scribble/logdir/prostate_deep_dropout_dif_exp'
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    input_path = sys_config.data_root
    recursion = args.RECURSION

#    apply_crf = args.apply_crf==1
    apply_crf=False
    
    recursion=0
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
    init_iteration = score_data(input_path, output_path, model_path, do_postprocessing=postprocess, exp_config=exp_config, recursion=recursion, apply_crf=apply_crf)

    metrics_acdc_simple_pro.main(path_gt, path_pred, path_eval, printing=printing)




