# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging

import argparse
import metrics_acdc_simple_crf
import time
from importlib.machinery import SourceFileLoader
import tensorflow as tf
from skimage import transform
from acdc_data import most_recent_recursion
import config.system as sys_config
import model as model
import utils
import image_utils
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#def apply_crf(seg_pred_probs,imgs):
#    
#    # Appearance parameters
##    a_sxy=160 # theta_alpha (refer website for equation terminology)
##    a_srgb= 3 # theta_beta
##    a_w1= 5 # weight term for bilateral term
##    # Gaussian smoothness term
##    g_sxy= 10 # theta_gamma
##    g_w2=30 # weight term for Gaussian smoothness
#
##    parameters= [6, 3, 0.01, 10, 2]
## 
##    a_w1, a_sxy, a_srgb = parameters[0],parameters[1],parameters[2]
##    g_w2, g_sxy = parameters[3],parameters[4]
#
##    a_sxy=0.5 # theta_alpha (refer website for equation terminology)
##    a_srgb= 1 # theta_beta
##    a_w1= 5 # weight term for bilateral term
##    # Gaussian smoothness term
##    g_sxy= 5 # theta_gamma
##    g_w2=30 # weight term for Gaussian smoothness
#    parameters= [6, 3, 0.01, 10, 2]
## 
#    a_w1, a_sxy, a_srgb = parameters[0],parameters[1],parameters[2]
#    g_w2, g_sxy = parameters[3],parameters[4]
#
##    a_sxy=0.5 # theta_alpha (refer website for equation terminology)
##    a_srgb= 1 # theta_beta
##    a_w1= 5 # weight term for bilateral term
##    # Gaussian smoothness term
##    g_sxy= 5 # theta_gamma
##    g_w2=30 # weight term for Gaussian smoothness
#
#    num_batch = seg_pred_probs.shape[0]
#    masks_out = np.zeros_like(imgs)
#    for k in range(num_batch):
#        seg_pred_prob = seg_pred_probs[k,:,:,:]
#        img = imgs[k,:,:,:]
#        seg_pred_prob_tmp = np.swapaxes(np.swapaxes( seg_pred_prob,0,2 ),1,2)# to above mentioned shape - you could use np.swapaxes to achieve this.
#    
#        unary = unary_from_softmax(seg_pred_prob_tmp)
#        
#        d = dcrf.DenseCRF2D(exp_config.image_size[0],exp_config.image_size[1], exp_config.nlabels)
#        d.setUnaryEnergy(unary)
#        
#        ###########################
#        #Calculate Bilateral term
#        ###########################
#        
#        ################################################
##        img_re = np.squeeze(img) #img_test_slice- 2D image containing intensity values
#        img_re = np.copy(img)
#        gaussian_pairwise_energy = create_pairwise_gaussian(sdims=(g_sxy,g_sxy), shape=img_re.shape[:2])
#        d.addPairwiseEnergy(gaussian_pairwise_energy, compat=g_w2)
#        
#        bilateral_pairwise_energy = create_pairwise_bilateral(sdims=(a_sxy,a_sxy), schan=(a_srgb,), img=img_re, chdim=2)
#        d.addPairwiseEnergy(bilateral_pairwise_energy, compat=a_w1) 
#
##        gaussian_pairwise_energy = create_pairwise_gaussian(shape=img_re.shape[:2])
##        d.addPairwiseEnergy(gaussian_pairwise_energy)
##        
##        bilateral_pairwise_energy = create_pairwise_bilateral(img=img_re, chdim=2)
##        d.addPairwiseEnergy(bilateral_pairwise_energy) 
#        ################################################
#        
#        ######################
#        # Inference 
#        ######################
#        # Run inference for 100 iterations
#        Q_final = d.inference(100)
#        
#        # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
#        crf_seg_soln = np.argmax(Q_final, axis=0)
#        
#        # Unfortunately, the DenseCRF flattens everything, so get it back into picture form (width,height).
#        crf_seg_soln = crf_seg_soln.reshape((exp_config.image_size[0],exp_config.image_size[1]))
#        masks_out[k,:,:,:] = np.copy(np.expand_dims(crf_seg_soln,2))
#    return masks_out

def apply_crf(seg_pred_probs,imgs):
    
    # Appearance parameters
#    a_sxy=160 # theta_alpha (refer website for equation terminology)
#    a_srgb= 3 # theta_beta
#    a_w1= 5 # weight term for bilateral term
#    # Gaussian smoothness term
#    g_sxy= 10 # theta_gamma
#    g_w2=30 # weight term for Gaussian smoothness

#    parameters= [6, 1, 0.01, 5, 10]
## 
#    a_w1, a_sxy, a_srgb = parameters[0],parameters[1],parameters[2]
#    g_w2, g_sxy = parameters[3],parameters[4]
    
    a_w1, a_sxy, a_srgb = 5, 2, 0.1
    g_w2, g_sxy = 10, 5
    num_batch = seg_pred_probs.shape[0]
    masks_out = np.zeros_like(imgs)
    for k in range(num_batch):
        seg_pred_prob = seg_pred_probs[k,:,:,:]
        print(seg_pred_prob.shape)
        img = imgs[k,:,:,:]
        seg_pred_prob_tmp = np.swapaxes(np.swapaxes( seg_pred_prob,0,2 ),1,2) # to above mentioned shape - you could use np.swapaxes to achieve this.
    
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
        proba = np.array(Q_final)
        
        out=np.zeros((exp_config.image_size[0],exp_config.image_size[1],5))
        
        for k in range(5):
            out[:,:,k] = proba[k,:].reshape((exp_config.image_size[0],exp_config.image_size[1]))
        
        out= np.copy(np.expand_dims(out,0))
    return out


def score_data(path_gt,path_pred,path_eval,input_folder, output_folder, model_path, exp_config, do_postprocessing=False, recursion=None):

    
    nx, ny = exp_config.image_size[:2]
    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl, exp_config.model_handle, exp_config.nlabels)
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

        a_w1, a_sxy, a_srgb = 5, 2, 0.1
        g_w2, g_sxy = 10, 5      
        parameters=[a_w1,a_sxy, a_srgb,g_w2,g_sxy]
        




        total_time = 0
        total_volumes = 0

        for folder in os.listdir(input_folder):

            folder_path = os.path.join(input_folder, folder)

            if os.path.isdir(folder_path):

                train_test = 'test' if (int(folder[-3:]) % 5 == 0) else 'train'

                if train_test == 'test':

                    infos = {}
                    for line in open(os.path.join(folder_path, 'Info.cfg')):
                        label, value = line.split(':')
                        infos[label] = value.rstrip('\n').lstrip(' ')

                    patient_id = folder.lstrip('patient')
                    ED_frame = int(infos['ED'])
                    ES_frame = int(infos['ES'])

                    for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                        logging.info(' ----- Doing image: -------------------------')
                        logging.info('Doing: %s' % file)
                        logging.info(' --------------------------------------------')

                        file_base = file.split('.nii.gz')[0]
                        file_mask = file_base + '_gt.nii.gz'

                        frame = int(file_base.split('frame')[-1])

                        img_dat = utils.load_nii(file)
                        mask_dat = utils.load_nii(file_mask)

                        img = img_dat[0].copy()
                        mask = mask_dat[0]

                        img = image_utils.normalise_image(img)



                        start_time = time.time()

                        if exp_config.data_mode == '2D':

                            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                            scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                            pixel_size[1] / exp_config.target_resolution[1])

                            predictions = []

                            for zz in range(img.shape[2]):

                                slice_img = np.squeeze(img[:,:,zz])
                                slice_rescaled = transform.rescale(slice_img,
                                                                   scale_vector,
                                                                   order=1,
                                                                   preserve_range=True,
                                                                   multichannel=False,
                                                                   mode='constant')

                                x, y = slice_rescaled.shape

                                x_s = (x - nx) // 2
                                y_s = (y - ny) // 2
                                x_c = (nx - x) // 2
                                y_c = (ny - y) // 2

                                # Crop section of image for prediction
                                if x > nx and y > ny:
                                    slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                                else:
                                    slice_cropped = np.zeros((nx,ny))
                                    if x <= nx and y > ny:
                                        slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                                    elif x > nx and y <= ny:
                                        slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                                    else:
                                        slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]


                                # GET PREDICTION
                                network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                                mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                                
                                logits_out = sess.run(tf.nn.softmax(logits_out))
                                logits_out=apply_crf(logits_out,network_input)
                                
                                prediction_cropped = np.squeeze(logits_out[0, ...])

                                # ASSEMBLE BACK THE SLICES
                                slice_predictions = np.zeros((x,y,num_channels))
                                # insert cropped region into original image again
                                if x > nx and y > ny:
                                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                                else:
                                    if x <= nx and y > ny:
                                        slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                                    elif x > nx and y <= ny:
                                        slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                                    else:
                                        slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]


#                                slice_predictions = np.zeros((x,y))
#                                # insert cropped region into original image again
#                                if x > nx and y > ny:
#                                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny] = prediction_cropped
#                                else:
#                                    if x <= nx and y > ny:
#                                        slice_predictions[:, y_s:y_s+ny] = prediction_cropped[x_c:x_c+ x, :]
#                                    elif x > nx and y <= ny:
#                                        slice_predictions[x_s:x_s + nx, :] = prediction_cropped[:, y_c:y_c + y]
#                                    else:
#                                        slice_predictions = prediction_cropped[x_c:x_c+ x, y_c:y_c + y]

                                # RESCALING ON THE LOGITS
                                prediction = transform.rescale(slice_predictions,
                                                               (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                               order=1,
                                                               preserve_range=True,
                                                               multichannel=False,
                                                               mode='constant')
                                prediction = np.uint8(np.argmax(prediction, axis=-1))
                                predictions.append(prediction)


                            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

                        elif exp_config.data_mode == '3D':


                            pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2],
                                          img_dat[2].structarr['pixdim'][3])

                            scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                            pixel_size[1] / exp_config.target_resolution[1],
                                            pixel_size[2] / exp_config.target_resolution[2])

                            vol_scaled = transform.rescale(img,
                                                           scale_vector,
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                            nz_max = exp_config.image_size[2]
                            slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)

                            nz_curr = vol_scaled.shape[2]
                            stack_from = (nz_max - nz_curr) // 2
                            stack_counter = stack_from

                            x, y, z = vol_scaled.shape

                            x_s = (x - nx) // 2
                            y_s = (y - ny) // 2
                            x_c = (nx - x) // 2
                            y_c = (ny - y) // 2

                            for zz in range(nz_curr):

                                slice_rescaled = vol_scaled[:, :, zz]

                                if x > nx and y > ny:
                                    slice_cropped = slice_rescaled[x_s:x_s + nx, y_s:y_s + ny]
                                else:
                                    slice_cropped = np.zeros((nx, ny))
                                    if x <= nx and y > ny:
                                        slice_cropped[x_c:x_c + x, :] = slice_rescaled[:, y_s:y_s + ny]
                                    elif x > nx and y <= ny:
                                        slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]

                                    else:
                                        slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice_rescaled[:, :]

                                slice_vol[:, :, stack_counter] = slice_cropped
                                stack_counter += 1

                            stack_to = stack_counter

                            network_input = np.float32(np.reshape(slice_vol, (1, nx, ny, nz_max, 1)))

                            start_time = time.time()
                            mask_out, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                            logging.info('Classified 3D: %f secs' % (time.time() - start_time))

                            prediction_nzs = mask_out[0, :, :, stack_from:stack_to]  # non-zero-slices

                            if not prediction_nzs.shape[2] == nz_curr:
                                raise ValueError('sizes mismatch')

                            # ASSEMBLE BACK THE SLICES
                            prediction_scaled = np.zeros(vol_scaled.shape)  # last dim is for logits classes

                            # insert cropped region into original image again
                            if x > nx and y > ny:
                                prediction_scaled[x_s:x_s + nx, y_s:y_s + ny, :] = prediction_nzs
                            else:
                                if x <= nx and y > ny:
                                    prediction_scaled[:, y_s:y_s + ny, :] = prediction_nzs[x_c:x_c + x, :, :]
                                elif x > nx and y <= ny:
                                    prediction_scaled[x_s:x_s + nx, :, :] = prediction_nzs[:, y_c:y_c + y, :]
                                else:
                                    prediction_scaled[:, :, :] = prediction_nzs[x_c:x_c + x, y_c:y_c + y, :]

                            logging.info('Prediction_scaled mean %f' % (np.mean(prediction_scaled)))

                            prediction = transform.resize(prediction_scaled,
                                                          (mask.shape[0], mask.shape[1], mask.shape[2], num_channels),
                                                          order=1,
                                                          preserve_range=True,
                                                          mode='constant')
                            prediction = np.argmax(prediction, axis=-1)
                            prediction_arr = np.asarray(prediction, dtype=np.uint8)


                        # This is the same for 2D and 3D again
                        if do_postprocessing:
                            prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

                        elapsed_time = time.time() - start_time
                        total_time += elapsed_time
                        total_volumes += 1

                        logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                        if frame == ED_frame:
                            frame_suffix = '_ED'
                        elif frame == ES_frame:
                            frame_suffix = '_ES'
                        else:
                            raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                             (frame, ED_frame, ES_frame))

                        # Save predicted mask
                        out_file_name = os.path.join(output_folder, 'prediction', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]
                        logging.info('saving to: %s' % out_file_name)
                        utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                        # Save GT image
                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % gt_file_name)
                        utils.save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                      'difference',
                                                      'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

                        # Save image data to the same folder for convenience
                        image_file_name = os.path.join(output_folder, 'image',
                                                'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % image_file_name)
                        utils.save_nii(image_file_name, img_dat[0], out_affine, out_header)

        logging.info('Average time per volume: %f' % (total_time/total_volumes))
        metrics_acdc_simple_crf.main(path_gt, path_pred, path_eval, parameters,  printing=printing)
    

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

    model_path = '/scratch_net/biwirender02/cany/scribble/logdir/heart_unet_crf_dif_exp'
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    # input_path = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617/'
    input_path = sys_config.data_root
    recursion = args.RECURSION
    recursion=1
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
    

    
#    values1 = [1,2,3,4,5]
#    values2 = [0.5,1,1.5,2]
#    values3 = [0.01,0.1,1,2]
#    values4 = [1,5,10,20,40]
#    values5 = [0.5,1,2,5,10]
#    
#    for k in range(len(values1)):
#        for m in range(len(values2)):
#            for n in range(len(values3)):
#                for o in range(len(values4)):
#                    for t in range(len(values5)):
                        
#                        parameters=[values1[k],values2[m],values3[n],values4[o],values5[t]]
    init_iteration = score_data(path_gt,path_pred,path_eval,input_path, output_path, model_path, do_postprocessing=postprocess, exp_config=exp_config, recursion=recursion)
                    
                        



