""" The main file to launch the inference of LVRV-net """

import sys
sys.path.append('..')

import os
import copy
import numpy as np
from itertools import izip
from scipy.misc import imresize
from PIL import Image as pil_image
import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef5_0,
    dice_coef5_1,
    dice_coef5_2,
    dice_coef5_3,
    dice_coef5,
    dice_coef5_loss,
    mean_variance_normalization5,
    elementwise_multiplication,
    keep_largest_components,
    touch_length_count,
    number_of_components,
    second_largest_component_count
)

from image2 import (
    array_to_img,
    ImageDataGenerator2
)

from data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc

from module_lvrv_net import net_module

import config



def predict_lvrv_net():

    code_path = config.code_dir
    
    fold = int(sys.argv[1])
    print('fold = {}'.format(fold))
    if fold == 0:
        mode = 'predict'
    elif fold in range(1,6):
        mode = 'val_predict'
    else:
        print('Incorrect fold')

    initial_lr = config.lvrv_net_initial_lr
    input_img_size = config.lvrv_net_input_img_size
    epochs = config.lvrv_net_epochs
    batch_size = 1

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=4)
    print('Loading model')

    
    model.load_weights(filepath=os.path.join(code_path, 'segmentation', 'model_lvrv_net_finetune_fold{}_epoch{}.h5'.format(str(fold), str(epochs).zfill(3))) )
    

    model.compile(optimizer=Adam(lr=initial_lr),loss=dice_coef5_loss, 
        metrics=[dice_coef5, dice_coef5_0, dice_coef5_1, dice_coef5_2, dice_coef5_3])

    print('This model has {} parameters'.format(model.count_params()) )


    seq_context_imgs, seq_context_segs, seq_imgs, seq_segs = data_lvrv_segmentation_propagation_acdc(mode = mode, fold = fold)
    


    predict_sequence = len(seq_imgs)

    # we create two instances with the same arguments for random transformation
    img_data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=0.,
                    width_shift_range=0., 
                    height_shift_range=0.,
                    shear_range=0., 
                    zoom_range=0.,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip= False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication

    #########################
    # Generators for training
    print('Creating generators for prediction')
    image_context_datagen = ImageDataGenerator2(**img_data_gen_args)
    image_datagen = ImageDataGenerator2(**img_data_gen_args)
    mask_context_datagen = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_context_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    image_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_context_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)


    print('Start prediction')
    print('There will be {} sequences'.format(predict_sequence) )

    
    for i in range(predict_sequence):
        print('Sequence # {}'.format(i) )

        # The lists fot the sequence
        context_imgs = seq_context_imgs[i]
        context_segs = seq_context_segs[i]
        imgs = seq_imgs[i]
        segs = seq_segs[i]


        image_context_generator = image_context_datagen.flow_from_path_list(
            path_list=context_imgs,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)

        image_generator = image_datagen.flow_from_path_list(
            path_list=imgs,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)

        mask_context_generator = mask_context_datagen.flow_from_path_list(
            path_list=context_segs,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)


        # Combine generators into one which yields image and masks
        predict_generator = izip(image_context_generator, image_generator, mask_context_generator)

        
        img_size = pil_image.open(imgs[0]).size
        size = img_size[0]

        

        for j in range(len(imgs)):
            
            img_context, img, mask_context = next(predict_generator)
            masks = model.predict([img_context, img, mask_context], 
                batch_size=batch_size, verbose=0)

            masks = np.reshape(masks, newshape=(input_img_size, input_img_size, 4))
            masks_resized = np.zeros((size, size, 4))
            for c in range(4):
                masks_resized[:, :, c] = imresize(masks[:, :, c], (size, size), interp='bilinear')
            prediction_resized = np.argmax(masks_resized, axis=-1)
            prediction_resized = np.reshape(prediction_resized, newshape=(size, size, 1))

            # Check whether the prediction is successful
            have_lvc = (1 in prediction_resized)
            have_lvm = (2 in prediction_resized)
            lvc_touch_background_length = touch_length_count(prediction_resized, size, size, 1, 0)
            lvc_touch_lvm_length = touch_length_count(prediction_resized, size, size, 1, 2)
            lvc_touch_rvc_length = touch_length_count(prediction_resized, size, size, 1, 3)

            lvc_second_largest_component_count = second_largest_component_count(prediction_resized, 1)
            lvm_second_largest_component_count = second_largest_component_count(prediction_resized, 2)
            rvc_second_largest_component_count = second_largest_component_count(prediction_resized, 3)

            
            success = have_lvm and \
                ((lvc_touch_background_length + lvc_touch_rvc_length) <= 0.5 * lvc_touch_lvm_length)


            
            if not success:
                prediction_resized = 0 * prediction_resized
                print('Unsuccessful segmentation for {}'.format(imgs[j]))
            else:
                prediction_resized = keep_largest_components(prediction_resized, keep_values=[1, 2, 3], values=[1, 2, 3])
            

            # save txt file
            prediction_path = segs[j]   
            prediction_txt_path = prediction_path.replace('.png', '.txt', 1)
            np.savetxt(prediction_txt_path, prediction_resized, fmt='%.6f')

            # save image
            prediction_img = array_to_img(prediction_resized * 50.0,
                                          data_format=None, 
                                          scale=False)
            prediction_img.save(prediction_path)
            


    K.clear_session()



if __name__ == '__main__':
    predict_lvrv_net()




