import numpy as np
import os

from helpers import myo_mask_max_min_mean_thickness
from image2 import load_img2

import config



def acdc_thickness():


    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    excluded_slice_ratio = config.excluded_slice_ratio
    seq_instants = config.acdc_seq_instants

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects + test_subjects 
    subjects = all_subjects



    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]


    pixel_file = os.path.join(code_dir, 'acdc_info', 'acdc_pixel_size.txt')
    with open(pixel_file) as p_file:
        pixel_size_info = p_file.readlines()

    pixel_size_info = [x.strip() for x in pixel_size_info]
    pixel_size_info = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in pixel_size_info]


    base_file = os.path.join(code_dir, 'acdc_info', 'acdc_base.txt')
    with open(base_file) as b_file:
        base_info = b_file.readlines()

    base_info = [x.strip() for x in base_info]
    base_info = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in base_info]



    zfill_num = 2
    img_size = config.apparentflow_net_input_img_size
    shape = (img_size, img_size ,2)
    


    thickness_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_thickness.txt'), 'w')

    for subject in subjects:
        print('\n'+subject)
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        slices = int([x for x in subject_info if x[0] == subject][0][5])
        base_slice =  int([x for x in base_info if x[0] == subject][0][1])
        apex_slice =  int([x for x in base_info if x[0] == subject][0][2])
        es_base_slice =  int([x for x in base_info if x[0] == subject][0][3])
        es_apex_slice =  int([x for x in base_info if x[0] == subject][0][4])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        bsa = [x for x in subject_info if x[0] == subject][0][8]
        pixel_size = [x for x in pixel_size_info if x[0] == subject][0][3]
        slice_thickness = [x for x in pixel_size_info if x[0] == subject][0][5]

        subject_dir = data_dir.format(subject)
        folder = subject_dir + '/predict_2D/'

        normalize_term = pixel_size / (bsa**(1.0/2))

        idx_range = [(int(round(float(instants) * t / seq_instants)) + ed_instant) % instants for t in range(0, seq_instants)]

    

        start_slice = base_slice
        end_slice = apex_slice + 1

        es_start_slice = es_base_slice
        es_end_slice = es_apex_slice + 1  


        ed_max = 0.0
        ed_min = 10000.0
        ed_sum = 0.0
        ed_used_slice_count = 0
        es_max = 0.0
        es_min = 10000.0
        es_sum = 0.0
        es_used_slice_count = 0

        for slice_idx in range(start_slice, end_slice):
            mask_file = folder + 'predict_lvrv2_{}_{}.png'.format(str(slice_idx).zfill(zfill_num), str(ed_instant).zfill(zfill_num))
            mask = load_img2(mask_file, grayscale=True, 
                         target_size=(shape[0], 
                         shape[1]),
                         pad_to_square=True, resize_mode='nearest')
            ed_max_thick, ed_min_thick, ed_mean_thick = myo_mask_max_min_mean_thickness(np.array(mask)/50.0)
            if ed_max_thick >= 0.0:
                ed_max = max(ed_max, ed_max_thick)
                ed_min = min(ed_min, ed_min_thick)
                ed_sum += ed_mean_thick
                ed_used_slice_count += 1
        ed_mean = ed_sum / ed_used_slice_count


        for slice_idx in range(es_start_slice, es_end_slice):
            es_mask_file = folder + 'predict_lvrv2_{}_{}.png'.format(str(slice_idx).zfill(zfill_num), str(es_instant).zfill(zfill_num))
            es_mask = load_img2(es_mask_file, grayscale=True, 
                         target_size=(shape[0], 
                         shape[1]),
                         pad_to_square=True, resize_mode='nearest')
            es_max_thick, es_min_thick, es_mean_thick = myo_mask_max_min_mean_thickness(np.array(es_mask)/50.0)
            if es_max_thick >= 0.0:
                es_max = max(es_max, es_max_thick)
                es_min = min(es_min, es_min_thick)
                es_sum += es_mean_thick
                es_used_slice_count += 1
        es_mean = es_sum / es_used_slice_count 

        ed_max *= pixel_size
        ed_min *= pixel_size
        es_max *= pixel_size
        es_min *= pixel_size
        ed_mean *= pixel_size
        es_mean *= pixel_size

        print(ed_max, ed_min, es_max, es_min, ed_mean, es_mean)

        
        written = '{} {} {} {} {} {} {}\n'.format(subject, ed_max, ed_min, es_max, es_min, ed_mean, es_mean)
        thickness_info.write(written)

    thickness_info.close()
    







if __name__ == '__main__':
    acdc_thickness()










        
