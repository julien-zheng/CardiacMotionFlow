
import sys
sys.path.append('..')

import os
import math
import numpy as np
import nibabel as nib

import config

def acdc_info():
    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    train_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects

    bsa_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_info.txt'), 'w')

    for subject in train_subjects:
        print(subject)
        subject_dir = data_dir.format(subject)
        subject_info_file = os.path.join(subject_dir, 'Info.cfg')
        with open(subject_info_file) as s_file:
            subject_info = s_file.readlines()

        subject_info = [x.strip() for x in subject_info]
        ED = int(subject_info[0][4:]) - 1
        ES = int(subject_info[1][4:]) - 1
        group = subject_info[2][7:]
        height = float(subject_info[3][8:])
        num_frame = int(subject_info[4][9:])
        weight = float(subject_info[5][8:])

        bsa = math.sqrt(weight * height / 3600)

        sa_zip_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        img = nib.load(sa_zip_file)
        data = img.get_data()
        data_np = np.array(data)
        slices = data.shape[2]
        

        written = '{} {} {} {} {} {} {} {} {}\n'.format(subject, group, num_frame, ED, ES, slices, height, weight, bsa)


        bsa_info.write(written)

    
    for subject in test_subjects:
        print(subject)
        subject_dir = data_dir.format(subject)
        subject_info_file = os.path.join(subject_dir, 'Info.cfg')
        with open(subject_info_file) as s_file:
            subject_info = s_file.readlines()

        subject_info = [x.strip() for x in subject_info]
        ED = int(subject_info[0][4:]) - 1
        ES = int(subject_info[1][4:]) - 1
        group = 'TEST'
        height = float(subject_info[2][8:])
        num_frame = int(subject_info[3][9:])
        weight = float(subject_info[4][8:])

        bsa = math.sqrt(weight * height / 3600)

        sa_zip_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        img = nib.load(sa_zip_file)
        data = img.get_data()
        data_np = np.array(data)
        slices = data.shape[2]
        

        written = '{} {} {} {} {} {} {} {} {}\n'.format(subject, group, num_frame, ED, ES, slices, height, weight, bsa)


        bsa_info.write(written)

    bsa_info.close()

if __name__ == '__main__':
    acdc_info()

