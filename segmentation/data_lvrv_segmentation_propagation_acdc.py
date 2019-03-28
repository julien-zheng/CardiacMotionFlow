""" A function to generate the lists of files for LVRV-net inference """

import sys
sys.path.append('..')

import os
import math

import config

def data_lvrv_segmentation_propagation_acdc(mode='all', fold = 1):

    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    excluded_slice_ratio = config.excluded_slice_ratio

    seq_instants = config.acdc_seq_instants

    
    all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects
    

    if mode == 'all':
        subjects = all_subjects
    elif mode == 'train':
        subjects = [x for i,x in enumerate(all_subjects) if (i % 5) != (fold % 5)]
    elif mode == 'val' or mode == 'val_predict':
        subjects = [x for i,x in enumerate(all_subjects) if (i % 5) == (fold % 5)]
    elif mode == 'predict':
        subjects = test_subjects
    else:
        print('Incorrect mode')

    print(subjects)



    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]


    print('There will be {} used subjects'.format(len(subjects)) ) 


    seq_context_imgs = []
    seq_context_segs = []
    seq_imgs = []
    seq_segs = []

    seq_context_imgs_no_group = []
    seq_context_segs_no_group = []
    seq_imgs_no_group = []
    seq_segs_no_group = []


    for subject in subjects:
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        slices = int([x for x in subject_info if x[0] == subject][0][5])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        subject_dir = data_dir.format(subject)

        start_slice = 0
        end_slice = slices 

        if not os.path.exists(os.path.join(subject_dir, 'predict_2D')):
            os.makedirs(os.path.join(subject_dir, 'predict_2D'))
        

        for t in [ed_instant, es_instant]:
            context_imgs = []
            context_segs = []
            imgs = []
            segs = []
            
            for i in range(start_slice, end_slice):
                if i == start_slice:
                    i_minus = -1
                else:
                    i_minus = i - 1

                 
                context_img = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )
                if mode in ['all', 'train', 'val']:    
                    context_seg = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )
                elif mode in ['predict', 'val_predict']:
                    context_seg = os.path.join(subject_dir, 'predict_2D', 'predict_lvrv2_{}_{}.png'.format(str(i_minus).zfill(2), str(t).zfill(2)) )



                img = os.path.join(subject_dir, 'crop_2D', 'crop_2D_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
                if mode in ['all', 'train', 'val']:
                    seg = os.path.join(subject_dir, 'crop_2D', 'crop_2D_gt_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )
                elif mode in ['predict', 'val_predict']:
                    seg = os.path.join(subject_dir, 'predict_2D', 'predict_lvrv2_{}_{}.png'.format(str(i).zfill(2), str(t).zfill(2)) )

                seq_context_imgs_no_group.append(context_img)
                seq_context_segs_no_group.append(context_seg)
                seq_imgs_no_group.append(img)
                seq_segs_no_group.append(seg)


                context_imgs.append(context_img)
                context_segs.append(context_seg)
                imgs.append(img)
                segs.append(seg)

            seq_context_imgs.append(context_imgs)
            seq_context_segs.append(context_segs)
            seq_imgs.append(imgs)
            seq_segs.append(segs)
                

    if mode in ['all', 'train', 'val']:
        return seq_context_imgs_no_group, seq_context_segs_no_group, seq_imgs_no_group, seq_segs_no_group
    elif mode in ['predict', 'val_predict']:
        return seq_context_imgs, seq_context_segs, seq_imgs, seq_segs
        

