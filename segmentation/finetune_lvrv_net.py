""" The main file to launch the finetuning of LVRV-net """

import sys
sys.path.append('..')

import os
import copy
import math
import numpy as np
from itertools import izip
import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef5,
    dice_coef5_loss,
    dice_coef5_0,
    dice_coef5_1,
    dice_coef5_2,
    dice_coef5_3,
    mean_variance_normalization5,
    elementwise_multiplication
)

from image2 import ImageDataGenerator2


from data_lvrv_segmentation_propagation_acdc import data_lvrv_segmentation_propagation_acdc

from module_lvrv_net import net_module

import config

save_period = 10000

def finetune_lvrv_net():

    code_path = config.code_dir
    
    fold = int(sys.argv[1])
    print('fold = {}'.format(fold))
    if fold == 0:
        mode_train = 'all'
        mode_val = 'all'
    elif fold in range(1,6):
        mode_train = 'train'
        mode_val = 'val'
    else:
        print('Incorrect fold')

    initial_lr = config.lvrv_net_initial_lr
    decay_rate = config.lvrv_net_decay_rate
    batch_size = config.lvrv_net_batch_size
    input_img_size = config.lvrv_net_input_img_size
    epochs = config.lvrv_net_epochs

    current_epoch = 80
    new_start_epoch = current_epoch

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=4)
    
    # Finetune
    print('Loading model')
    model.load_weights(filepath=os.path.join(code_path, 'segmentation', 'model_lvrv_net_epoch{}.h5'.format(str(current_epoch).zfill(3))) )

    model.compile(optimizer=Adam(lr=initial_lr), loss=dice_coef5_loss, 
        metrics=[dice_coef5, dice_coef5_0, dice_coef5_1, dice_coef5_2, dice_coef5_3])
    

    print('This model has {} parameters'.format(model.count_params()) )


    # Load data lists
    train_img_list0, train_gt_list0, train_img_list1, train_gt_list1 = \
    data_lvrv_segmentation_propagation_acdc(mode = mode_train, fold = fold)

    test_img_list0, test_gt_list0, test_img_list1, test_gt_list1 = \
    data_lvrv_segmentation_propagation_acdc(mode = mode_val, fold = fold)

    training_sample = len(train_img_list0)

    # we create two instances with the same arguments for random transformation
    img_data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=180.,
                    width_shift_range=0.15, 
                    height_shift_range=0.15,
                    shear_range=0., 
                    zoom_range=0.15,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip=True, 
                    vertical_flip=True,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication


    #########################
    # Generators for training
    print('Creating generators for training')
    image_datagen0 = ImageDataGenerator2(**img_data_gen_args)
    image_datagen1 = ImageDataGenerator2(**img_data_gen_args)
    mask_datagen0 = ImageDataGenerator2(**mask_data_gen_args)
    mask_datagen1 = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    image_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)

    image_generator0 = image_datagen0.flow_from_path_list(
        path_list=train_img_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='img0_',
        save_format='png',
        save_period=save_period,
        follow_links=False)

    image_generator1 = image_datagen1.flow_from_path_list(
        path_list=train_img_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='img1_',
        save_format='png',
        save_period=save_period,
        follow_links=False)

    mask_generator0 = mask_datagen0.flow_from_path_list(
        path_list=train_gt_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='mask0_',
        save_format='png',
        save_period=save_period,
        follow_links=False)

    mask_generator1 = mask_datagen1.flow_from_path_list(
        path_list=train_gt_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='mask1_',
        save_format='png',
        save_period=save_period,
        follow_links=False)

    # Combine generators into one which yields image and masks
    train_generator = izip(image_generator0, image_generator1, 
                           mask_generator0, mask_generator1)

    
    ###########################
    # Generators for validation
    print('Creating generators for validation')
    val_image_datagen0 = ImageDataGenerator2(**img_data_gen_args)
    val_image_datagen1 = ImageDataGenerator2(**img_data_gen_args)
    val_mask_datagen0 = ImageDataGenerator2(**mask_data_gen_args)
    val_mask_datagen1 = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    val_seed = 2
    val_image_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_image_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_mask_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_mask_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)

    val_image_generator0 = val_image_datagen0.flow_from_path_list(
        path_list=test_img_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='img0_',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_image_generator1 = val_image_datagen1.flow_from_path_list(
        path_list=test_img_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='img1_',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_mask_generator0 = val_mask_datagen0.flow_from_path_list(
        path_list=test_gt_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest',
        histogram_based_preprocessing=False,
        clahe=False, 
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='mask0_',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_mask_generator1 = val_mask_datagen1.flow_from_path_list(
        path_list=test_gt_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='mask1_',
        save_format='png',
        save_period=1,
        follow_links=False)


    # Combine generators into one which yields image and masks
    validation_generator = izip(val_image_generator0, val_image_generator1, 
                                val_mask_generator0, val_mask_generator1)


    ###############
    # Train the model
    print('Start training')
    steps = int(math.ceil(float(training_sample) / batch_size))
    print('There will be {} epochs with {} steps in each epoch'.format(epochs, steps) )


    total_step = 0
    for epoch in range(new_start_epoch + 1, epochs + 1):
        print('\n\n##########\nEpoch {}\n##########'.format(epoch) )

        for step in range(steps):
            print('\n****** Epoch {} Step {} ******'.format(epoch, step) )
            batch_img0, batch_img1, batch_mask0, batch_mask1 = next(train_generator)
            print(model.train_on_batch([batch_img0, batch_img1, batch_mask0], 
                                       batch_mask1, sample_weight=None, class_weight=None))

            
            # perform test
            if (total_step % save_period == 0):
                val_batch_img0, val_batch_img1, \
                val_batch_mask0, val_batch_mask1 = next(validation_generator)
                print('test:')
                print(model.test_on_batch([val_batch_img0, val_batch_img1, val_batch_mask0], 
                                          val_batch_mask1, sample_weight=None))

            total_step += 1


        # adjust learning rate
        if (epoch % 10 == 0):
            old_lr = float(K.get_value(model.optimizer.lr))
            new_lr = initial_lr * (decay_rate**(epoch//10))
            K.set_value(model.optimizer.lr, new_lr)
            print('learning rate is reset to %.8f' % (new_lr))

        # save the weights of the model
        if (epoch % 1000 == 0):
            model.save_weights(os.path.join(code_path, 'segmentation', 'model_lvrv_net_finetune_fold{}_epoch{}.h5'.format(str(fold), str(epoch).zfill(3)) ) )


    print('Training is done!')


if __name__ == '__main__':
    finetune_lvrv_net()




