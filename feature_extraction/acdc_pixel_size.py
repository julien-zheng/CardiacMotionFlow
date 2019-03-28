
import sys
sys.path.append('..')

import os
import math
from scipy import misc
import nibabel as nib

import config

def acdc_pixel_size():
    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    new_img_size = config.apparentflow_net_input_img_size

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects + test_subjects 

    pixel_size_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_pixel_size.txt'), 'w')

    for subject in all_subjects:
        print(subject)
        subject_dir = data_dir.format(subject)
        subject_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        subject_img = nib.load(subject_file)
        header = subject_img.header
        #print(header.get_zooms())
        pixel_size = header.get_zooms()[0]
        slice_thickness = header.get_zooms()[2]

        predict_dir = os.path.join(subject_dir, 'predict_2D')
        a_prediction_file = ''
        for f in os.listdir(predict_dir):
            if f.startswith('predict_lvrv2_') and f.endswith('png'):
                a_prediction_file = f
                break
        a_prediction_file_full = os.path.join(predict_dir, a_prediction_file)
        a_prediction = misc.imread(a_prediction_file_full)
        #print(a_prediction.shape)
        roi_size = a_prediction.shape[0]

        new_pixel_size = pixel_size * roi_size / new_img_size

        written = '{} {} {} {} {} {}\n'.format(subject, pixel_size, roi_size, new_pixel_size, new_img_size, slice_thickness)

        pixel_size_info.write(written)

    pixel_size_info.close()

if __name__ == '__main__':
    acdc_pixel_size()




