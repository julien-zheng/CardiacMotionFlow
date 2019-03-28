import numpy as np
import os

from helpers import volume_calculation_given_slice_area
from image2 import load_img2

import config



def acdc_volume():


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
    


    volume_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_volume.txt'), 'w')

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

    

        start_slice = 0
        end_slice = slices    


        lv_area_ed = []
        lv_area_es = []
        lvm_area_ed = []
        lvm_area_es = []
        rv_area_ed = []
        rv_area_es = []

        for slice_idx in range(start_slice, end_slice):
            #print('slice #{}'.format(slice_idx))


            # Get the segmentation
            #print('Get the segmentation')
            seg = np.zeros((1, shape[0], shape[1], 0))
            for idx in [ed_instant, es_instant]:
                seg_file = folder + 'predict_lvrv2_{}_{}.png'.format(str(slice_idx).zfill(zfill_num), str(idx).zfill(zfill_num))

                seg_idx = load_img2(seg_file, grayscale=True, 
                         target_size=(shape[0], 
                         shape[1]),
                         pad_to_square=True, resize_mode='nearest')
                seg_idx = np.reshape(np.array(seg_idx)/50.0, (1, shape[0], shape[1], 1))
                seg = np.concatenate((seg, seg_idx), axis=-1)
        
            seg_rv = np.where(np.equal(seg, 3.0 * np.ones_like(seg)), 
                np.ones_like(seg), np.zeros_like(seg))
            seg_rv_area = np.sum(seg_rv, axis=(1,2))
            seg_rv_area *= ((normalize_term**2)/1000)

            seg_lv = np.where(np.equal(seg, 1.0 * np.ones_like(seg)), 
                np.ones_like(seg), np.zeros_like(seg))
            seg_lv_area = np.sum(seg_lv, axis=(1,2))
            seg_lv_area *= ((normalize_term**2)/1000)

            seg_lvm = np.where(np.equal(seg, 2.0 * np.ones_like(seg)), 
                np.ones_like(seg), np.zeros_like(seg))
            seg_lvm_area = np.sum(seg_lvm, axis=(1,2))
            seg_lvm_area *= ((normalize_term**2)/1000)

            
            lv_area_ed.append(seg_lv_area[0, 0])
            lv_area_es.append(seg_lv_area[0, 1])
            lvm_area_ed.append(seg_lvm_area[0, 0])
            lvm_area_es.append(seg_lvm_area[0, 1])
            rv_area_ed.append(seg_rv_area[0, 0])
            rv_area_es.append(seg_rv_area[0, 1])

        lv_volume_ed = volume_calculation_given_slice_area(lv_area_ed, slice_thickness)
        lv_volume_es = volume_calculation_given_slice_area(lv_area_es, slice_thickness)
        lvm_volume_ed = volume_calculation_given_slice_area(lvm_area_ed, slice_thickness)
        lvm_volume_es = volume_calculation_given_slice_area(lvm_area_es, slice_thickness)
        rv_volume_ed = volume_calculation_given_slice_area(rv_area_ed, slice_thickness)
        rv_volume_es = volume_calculation_given_slice_area(rv_area_es, slice_thickness)


        lv_ratio = 1.0 - lv_volume_es / lv_volume_ed
        lvm_ratio = lvm_volume_es / lvm_volume_ed
        rv_ratio = 1.0 - rv_volume_es / rv_volume_ed
        lvrv_ratio = rv_volume_ed / lv_volume_ed
        lvmrv_ratio = rv_volume_ed / (lv_volume_ed + lvm_volume_ed)
        lvmlv_ratio = lvm_volume_ed / lv_volume_ed
        print(lv_volume_ed, rv_volume_ed, lvm_volume_ed, lv_ratio, rv_ratio, lvm_ratio, lvrv_ratio, lvmrv_ratio, lvmlv_ratio)
        

        
        written = '{} {} {} {} {} {} {} {} {} {}\n'.format(subject, lv_volume_ed, rv_volume_ed, lvm_volume_ed, lv_ratio, rv_ratio, lvm_ratio, lvrv_ratio, lvmrv_ratio, lvmlv_ratio)
        volume_info.write(written)

    volume_info.close()
    



if __name__ == '__main__':
    acdc_volume()



        
