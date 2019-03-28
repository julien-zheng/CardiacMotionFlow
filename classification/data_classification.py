""" A function to generate the lists of features for classification """

import sys
sys.path.append('..')

import os

import config

def data_classification(mode='all', fold = 1, data_class_num = 5, normalization=True):

    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects
    

    if data_class_num == 2:
        all_subjects = infarct_subjects + normal_subjects
    elif data_class_num == 3:
        all_subjects = dilated_subjects + infarct_subjects + normal_subjects
    elif data_class_num == 4:
        all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects
    elif data_class_num == 5:
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

    #print(subjects)

    excluded_slice_ratio = config.excluded_slice_ratio

    seq_instants = config.acdc_seq_instants

    zfill_num = 2



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
    base_info = [ [y.split()[0]] + [int(z) for z in y.split()[1:]] for y in base_info]


    volume_file = os.path.join(code_dir, 'acdc_info', 'acdc_volume.txt')

    with open(volume_file) as v_file:
        volume_info = v_file.readlines()

    volume_info = [x.strip() for x in volume_info]
    volume_info = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in volume_info]


    thickness_file = os.path.join(code_dir, 'acdc_info', 'acdc_thickness.txt')

    with open(thickness_file) as t_file:
        thickness_info = t_file.readlines()

    thickness_info = [x.strip() for x in thickness_info]
    thickness_info = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in thickness_info]


    motion_file = os.path.join(code_dir, 'acdc_info', 'acdc_motion_index.txt')

    with open(motion_file) as m_file:
        motion_info = m_file.readlines()

    motion_info = [x.strip() for x in motion_info]
    motion_info = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in motion_info]




    #print('There will be {} used subjects'.format(len(subjects)) ) 

    list_subject_idx = []
    list_lv_volume = []
    list_rv_volume = []
    list_lv_ratio = []
    list_rv_ratio = []
    list_lvmrv_ratio = []
    list_lvmlv_ratio = []
    list_lvmlv_mass = []
    list_thickness = []
    list_thickness_diff = []
    list_asyn_radius = []
    list_asyn_thickness = []
    list_gt = [] 



    for subject in subjects:
        subject_idx = int(subject[-3:])
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        slices = int([x for x in subject_info if x[0] == subject][0][5])
        base_slice =  int([x for x in base_info if x[0] == subject][0][1])
        apex_slice =  int([x for x in base_info if x[0] == subject][0][2])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        bsa = [x for x in subject_info if x[0] == subject][0][8]
        pixel_size = [x for x in pixel_size_info if x[0] == subject][0][3]
        slice_thickness = [x for x in pixel_size_info if x[0] == subject][0][5]

        subject_dir = data_dir.format(subject)
        folder = subject_dir + '/predict_2D/'

        slice_range = range(base_slice + int(round((apex_slice + 1 - base_slice)*excluded_slice_ratio)), apex_slice + 1 - int(round((apex_slice + 1 - base_slice)*2*excluded_slice_ratio)))

        normalize_term = pixel_size / (bsa**(1.0/2))

        lv_volume = [x for x in volume_info if x[0] == subject][0][1]
        rv_volume = [x for x in volume_info if x[0] == subject][0][2]
        lvm_volume = [x for x in volume_info if x[0] == subject][0][3]
        lv_ratio = [x for x in volume_info if x[0] == subject][0][4]
        rv_ratio = [x for x in volume_info if x[0] == subject][0][5]
        lvm_ratio = [x for x in volume_info if x[0] == subject][0][6]
        lvmrv_ratio = [x for x in volume_info if x[0] == subject][0][8]
        lvmlv_ratio = [x for x in volume_info if x[0] == subject][0][9]
        thickness = [x for x in thickness_info if x[0] == subject][0][1]
        es_thickness = [x for x in thickness_info if x[0] == subject][0][3]

        if not normalization:
            lv_volume *= bsa
            rv_volume *= bsa
            lvm_volume *= bsa

        lvmlv_mass = 1.06 * (lv_volume + lvm_volume)
        

        lv_volume_es = (1.0 - lv_ratio) * lv_volume
        rv_volume_es = (1.0 - rv_ratio) * rv_volume
        lvm_volume_es = lvm_ratio * lvm_volume
        lvmlv_ratio_es = lvm_volume_es / lv_volume_es 
        lvmrv_ratio_es = rv_volume_es / (lv_volume_es + lvm_volume_es)
        


        asyn_radius = [x for x in motion_info if x[0] == subject][0][1]
        asyn_thickness = [x for x in motion_info if x[0] == subject][0][2]
        thickness_diff = [x for x in motion_info if x[0] == subject][0][3]
        

        if subject in dilated_subjects:
            gt = 0
        elif subject in hypertrophic_subjects:
            gt = 1
        elif subject in infarct_subjects:
            gt = 2
        elif subject in normal_subjects:
            gt = 3
        elif subject in rv_subjects:
            gt = 4
        elif subject in test_subjects:
            gt = -1




        list_subject_idx.append(subject_idx)
        list_lv_volume.append(lv_volume_es)
        list_rv_volume.append(rv_volume)
        list_lv_ratio.append(lv_ratio)
        list_rv_ratio.append(rv_ratio)
        list_lvmrv_ratio.append(lvmrv_ratio)
        list_lvmlv_ratio.append(lvmlv_ratio)
        list_lvmlv_mass.append(lvmlv_mass)
        list_thickness.append(thickness)
        list_thickness_diff.append(thickness_diff)
        list_asyn_radius.append(asyn_radius)
        list_asyn_thickness.append(asyn_thickness)
        list_gt.append(gt)


    return list_subject_idx, list_lv_volume, list_rv_volume, list_lv_ratio, list_rv_ratio, list_lvmrv_ratio, list_lvmlv_ratio, list_lvmlv_mass, list_thickness, list_thickness_diff, list_asyn_radius, list_asyn_thickness, list_gt



