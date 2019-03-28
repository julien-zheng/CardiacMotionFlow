
import sys
sys.path.append('..')

import os
import math
import numpy as np
from PIL import Image

import config

def acdc_gt_base():
    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    all_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects



    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]


    base_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_gt_base.txt'), 'w')

    for subject in all_subjects:
        subject_dir = data_dir.format(subject)
        subject_predict_dir = os.path.join(subject_dir, 'crop_2D')
        subject_predict_file = os.path.join(subject_predict_dir, 'crop_2D_gt_{}_{}.png')

        
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        slices = int([x for x in subject_info if x[0] == subject][0][5])

        base_slice = 0
        have_rv = False
        for i in range(slices):
            img_file = subject_predict_file.format(str(i).zfill(2), str(ed_instant).zfill(2))
            img = Image.open(img_file)
            img.load()
            data = np.array(img)
            if 150 in data:
                base_slice = i
                have_rv = True
                break
        if not have_rv:
            for i in range(slices):
                img_file = subject_predict_file.format(str(i).zfill(2), str(ed_instant).zfill(2))
                img = Image.open(img_file)
                img.load()
                data = np.array(img)
                if 50 in data:
                    base_slice = i
                    break

        apex_slice = slices-1
        for j in range(slices-1, -1, -1):
            img_file = subject_predict_file.format(str(j).zfill(2), str(ed_instant).zfill(2))
            img = Image.open(img_file)
            img.load()
            data = np.array(img)
            if 50 in data:
                apex_slice = j
                break


        es_base_slice = 0
        have_rv = False
        for i in range(slices):
            img_file = subject_predict_file.format(str(i).zfill(2), str(es_instant).zfill(2))
            img = Image.open(img_file)
            img.load()
            data = np.array(img)
            if 150 in data:
                es_base_slice = i
                have_rv = True
                break
        if not have_rv:
            for i in range(slices):
                img_file = subject_predict_file.format(str(i).zfill(2), str(es_instant).zfill(2))
                img = Image.open(img_file)
                img.load()
                data = np.array(img)
                if 50 in data:
                    es_base_slice = i
                    break

        es_apex_slice = slices-1
        for j in range(slices-1, -1, -1):
            img_file = subject_predict_file.format(str(j).zfill(2), str(es_instant).zfill(2))
            img = Image.open(img_file)
            img.load()
            data = np.array(img)
            if 50 in data:
                apex_slice = j
                break

        print(subject, base_slice, apex_slice, es_base_slice, es_apex_slice)
        written = '{} {} {} {} {}\n'.format(subject, base_slice, apex_slice, es_base_slice, es_apex_slice)
        base_info.write(written)

    base_info.close()

if __name__ == '__main__':
    acdc_gt_base()

