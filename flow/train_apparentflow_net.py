""" The main file to launch the training of ApparentFlow-net """

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
    flow_warped_img_comparison_loss,
    flow_warped_gt_comparison_dice_loss,
    flow_diffeomorphism_loss,
    flow_combined_loss3,
    mean_variance_normalization5,
    elementwise_multiplication
)

from image2 import ImageDataGenerator2



from data_apparentflow import data_apparentflow

from module_apparentflow_net import net_module

import config

save_period = 10000

def train_apparentflow_net():

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


    initial_lr = config.apparentflow_net_initial_lr
    decay_rate = config.apparentflow_net_decay_rate
    batch_size = config.apparentflow_net_batch_size
    input_img_size = config.apparentflow_net_input_img_size
    epochs = config.apparentflow_net_epochs

    current_epoch = 0
    new_start_epoch = current_epoch

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=2)
    # Train from scratch
    if current_epoch == 0:
        print('Building model')
    # Finetune
    else:
        print('Loading model')
        model.load_weights(filepath=os.path.join(code_path, 'flow', 'model_apparentflow_net_fold{}_epoch{}.h5'.format(str(fold), str(current_epoch).zfill(3))) )

    
    model.compile(optimizer=Adam(lr=initial_lr), loss=flow_combined_loss3, 
        metrics=[flow_warped_img_comparison_loss, flow_warped_gt_comparison_dice_loss, flow_diffeomorphism_loss])
    

    print('This model has {} parameters'.format(model.count_params()) )

    # Load data lists
    train_img_list0, train_img_list1, train_gt_list0, train_gt_list1 = data_apparentflow(mode=mode_train, fold = fold)
    test_img_list0, test_img_list1, test_gt_list0, test_gt_list1 = data_apparentflow(mode=mode_val, fold = fold)
        
    train_sample = len(train_img_list0)
    val_sample = len(test_img_list0)

    train_img_list = [train_img_list0, train_img_list1, train_gt_list0, train_gt_list1]
    val_img_list = [test_img_list0, test_img_list1, test_gt_list0, test_gt_list1]

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
    seed = 1
    generators = []
    # The generators for the 2 inputs
    for k in range(0, 2):
        seg_datagen_k = ImageDataGenerator2(**img_data_gen_args)
        seg_datagen_k.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        seg_generator_k = seg_datagen_k.flow_from_path_list(
            path_list=train_img_list[k],
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
            save_prefix='',
            save_format='png',
            save_period=save_period,
            follow_links=False)
        generators.append(seg_generator_k)
    # The generators for the 2 masks
    for k in range(2, 4):
        seg_datagen_k = ImageDataGenerator2(**mask_data_gen_args)
        seg_datagen_k.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        seg_generator_k = seg_datagen_k.flow_from_path_list(
            path_list=train_img_list[k],
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
            save_prefix='',
            save_format='png',
            save_period=save_period,
            follow_links=False)
        generators.append(seg_generator_k)

    # Combine generators into one which yields image and masks
    train_generator = izip(*tuple(generators))

    
    ###########################
    # Generators for validation
    print('Creating generators for validation')
    val_seed = 2
    generators2 = []
    # The generators for the inputs
    for k in range(0, 2):
        seg_datagen_k = ImageDataGenerator2(**img_data_gen_args)
        seg_datagen_k.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        seg_generator_k = seg_datagen_k.flow_from_path_list(
            path_list=val_img_list[k],
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
            save_prefix='val_',
            save_format='png',
            save_period=save_period,
            follow_links=False)
        generators2.append(seg_generator_k)
    # The generators for the 2 masks
    for k in range(2, 4):
        seg_datagen_k = ImageDataGenerator2(**mask_data_gen_args)
        seg_datagen_k.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
        seg_generator_k = seg_datagen_k.flow_from_path_list(
            path_list=val_img_list[k],
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
            save_prefix='val_',
            save_format='png',
            save_period=save_period,
            follow_links=False)
        generators2.append(seg_generator_k)

    # Combine generators into one which yields image and masks
    val_generator = izip(*tuple(generators2))



    ###############
    # Train the model
    print('Start training')
    steps = int(math.ceil(float(train_sample) / batch_size))
    print('There will be {} epochs with {} steps in each epoch'.format(epochs, steps) )


    total_step = 0
    for epoch in range(new_start_epoch + 1, new_start_epoch + epochs + 1):
        print('\n\n##########\nEpoch {}\n##########'.format(epoch) )

        for step in range(steps):
            print('\n****** Epoch {} Step {} ******'.format(epoch, step) )
            train_batch = next(train_generator)
            
            print(model.train_on_batch([train_batch[0], train_batch[1]], 
                np.concatenate((train_batch[1], train_batch[0], train_batch[3], train_batch[2]), axis=-1), 
                sample_weight=None, class_weight=None))

            
            

            # perform test
            if (total_step % save_period == 0):
                val_batch = next(val_generator)
                print('test:')
                print(model.test_on_batch([val_batch[0], val_batch[1]], 
                    np.concatenate((val_batch[1], val_batch[0], val_batch[3], val_batch[2]), axis=-1), sample_weight=None))

                
            total_step += 1


        # adjust learning rate
        if (epoch % 10 == 0):
            old_lr = float(K.get_value(model.optimizer.lr))
            new_lr = initial_lr * (decay_rate**(epoch//10))
            K.set_value(model.optimizer.lr, new_lr)
            print("learning rate is reset to %.8f" % (new_lr))

        # save the model
        if (epoch % 50 == 0):
            model.save_weights(os.path.join(code_path, 'flow', 'model_apparentflow_net_fold{}_epoch{}.h5'.format(str(fold), str(epoch).zfill(3))) )



    print('Training is done!')


if __name__ == '__main__':
    train_apparentflow_net()




