""" A function to generate the lists of files for ApparentFlow-net """

import sys
sys.path.append('..')

import os
import math

import config

def data_apparentflow(mode='all', fold = 1):

    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects

    if mode == 'all':
        subjects = all_subjects
    elif mode == 'train':
        subjects = [x for i,x in enumerate(all_subjects) if (i % 5) != (fold % 5)]
    elif mode == 'val':
        subjects = [x for i,x in enumerate(all_subjects) if (i % 5) == (fold % 5)]
    elif mode == 'predict':
        subjects = test_subjects
    else:
        print('Incorrect mode')

    print(subjects)

    excluded_slice_ratio = config.excluded_slice_ratio

    seq_instants = config.acdc_seq_instants


    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]


    gt_base_file = os.path.join(code_dir, 'acdc_info', 'acdc_gt_base.txt')

    with open(gt_base_file) as g_file:
        gt_base_info = g_file.readlines()

    gt_base_info = [x.strip() for x in gt_base_info]
    gt_base_info = [ [y.split()[0]] + [int(z) for z in y.split()[1:]] for y in gt_base_info]


    print('There will be {} used subjects'.format(len(subjects)) ) 

    img_list0 = []
    img_list1 = []
    seg_list0 = []
    seg_list1 = []

    segmented_pair_count = 0
    unsegmented_pair_count = 0
    for subject in subjects:
        #print(subject)
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        slices = int([x for x in subject_info if x[0] == subject][0][5])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        subject_dir = data_dir.format(subject)


        if mode in ['test', 'predict']:
            start_slice = 0
            end_slice = slices
        else:
            base_slice =  int([x for x in gt_base_info if x[0] == subject][0][1])
            apex_slice =  int([x for x in gt_base_info if x[0] == subject][0][2])
            es_base_slice =  int([x for x in gt_base_info if x[0] == subject][0][3])
            es_apex_slice =  int([x for x in gt_base_info if x[0] == subject][0][4])
        
            # The start_slice is smaller than the end_slice
            start_slice = base_slice + int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
            end_slice = apex_slice + 1 - int(round((apex_slice + 1 - base_slice) * excluded_slice_ratio))
        

        for i in range(start_slice, end_slice):
            
            
            for t in range(0, instants):
                
                img0 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(ed_instant).zfill(2)) )
                img1 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
                if t == es_instant:
                    seg0 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(ed_instant).zfill(2)) )
                    seg1 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
                    segmented_pair_count += 1
                else:
                    seg0 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(-1).zfill(2)) )
                    seg1 = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(-1).zfill(2)) )
                    unsegmented_pair_count += 1
                img_list0.append(img0)
                img_list1.append(img1)
                seg_list0.append(seg0)
                seg_list1.append(seg1)

            

    print('pair count = {}'.format(len(img_list0)) )
    print('segmented_pair_count = {}'.format(segmented_pair_count), 'unsegmented_pair_count = {}'.format(unsegmented_pair_count))

    return img_list0, img_list1, seg_list0, seg_list1



