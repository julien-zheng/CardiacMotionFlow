""" A function to generate the lists of files for ROI-net inference"""

import sys
sys.path.append('..')

import os

import config

def data_roi_predict():
    data_dir = config.acdc_data_dir
    code_dir = config.code_dir



    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]

    
    
    
    predict_img_list = []
    predict_gt_list = []

    all_subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 151)]
    for subject in all_subjects:
        subject_dir = data_dir.format(subject)
        subject_predict_dir = os.path.join(subject_dir, 'mask_original_2D')
        if not os.path.exists(subject_predict_dir):
            os.makedirs(subject_predict_dir)
        #subject_predict_file = os.path.join(subject_predict_dir, 'mask_original_2D_{}_{}.png')


        
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        slices = int([x for x in subject_info if x[0] == subject][0][5])


        original_2D_path = os.path.join(subject_dir, 'original_2D')

        # Prediction on the ED stacks only
        used_instants = [ed_instant]
        
        for idx, t in enumerate(used_instants):
            for s in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))):
                s_t_image_file = os.path.join(original_2D_path, 'original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                # The adapted ground-truth
                s_t_image_gt_file = ''

                predict_img_list.append(s_t_image_file)
                predict_gt_list.append(s_t_image_gt_file)


    print('predict_image_count = {}'.format(len(predict_img_list)) )

    return predict_img_list, predict_gt_list





