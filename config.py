#!/usr/bin/env python


# Directory of the project
code_dir = '/home/qzheng/Programs/tensorflow/my_models/CardiacMotionFlow'


excluded_slice_ratio = 0.2



# For the ACDC data 
acdc_data_dir = '/data/asclepios/user/qzheng/Data/MICCAI2017_ACDC_Challenge2/{}'

acdc_dilated_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 21)]
acdc_hypertrophic_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(21, 41)]
acdc_infarct_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(41, 61)]
acdc_normal_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(61, 81)]
acdc_rv_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(81, 101)]
acdc_test_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(101, 151)]

acdc_seq_instants = 10



# ROI-net
roi_net_initial_lr = 1e-4
roi_net_decay_rate = 1.0
roi_net_batch_size = 16
roi_net_input_img_size = 128
roi_net_epochs = 50


# LVRV-net
lvrv_net_initial_lr = 1e-4
lvrv_net_decay_rate = 1.0
lvrv_net_batch_size = 16
lvrv_net_input_img_size = 192
lvrv_net_epochs = 1000


# ApparentFlow-net
apparentflow_net_initial_lr = 1e-4
apparentflow_net_decay_rate = 1.0
apparentflow_net_batch_size = 16
apparentflow_net_input_img_size = 128
apparentflow_net_epochs = 50





