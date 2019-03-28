""" The main file to launch ROI cropping according to the prediction of ROI-net """

import os
import sys
sys.path.append('..')

import numpy as np
import scipy
from scipy import ndimage
import math
import nibabel as nib
from PIL import Image

import multiprocessing.pool
from functools import partial

import config


# Auxiliary function
def determine_rectangle_roi(img_path):
    img = Image.open(img_path)
    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]

# Auxiliary function
def determine_rectangle_roi2(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    connected_components, num_connected_components = ndimage.label(img_array)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        single_component = connected_components * (connected_components == max_idx)
        img = Image.fromarray(single_component)

    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]


def change_array_values(array):
    output = array
    for u in range(output.shape[0]):
        for v in range(output.shape[1]):
            if output[u,v] == 1:
                output[u,v] = 3
            elif output[u,v] == 3:
                output[u,v] = 1
    return output


def crop_according_to_roi():
    # The ratio that determines the width of the margin
    pixel_margin_ratio = 0.3
    
    # If for a case there is non-zero pixels on the border of ROI, the case is stored in
    # this list for further examination. This list is eventually empty for UK Biobank cases.
    border_problem_subject = []



    data_dir = config.acdc_data_dir
    code_dir = config.code_dir

    dilated_subjects = config.acdc_dilated_subjects
    hypertrophic_subjects = config.acdc_hypertrophic_subjects
    infarct_subjects = config.acdc_infarct_subjects 
    normal_subjects = config.acdc_normal_subjects
    rv_subjects = config.acdc_rv_subjects
    test_subjects = config.acdc_test_subjects

    train_subjects = dilated_subjects + hypertrophic_subjects + infarct_subjects + normal_subjects + rv_subjects

    all_subjects = train_subjects + test_subjects



    info_file = os.path.join(code_dir, 'acdc_info', 'acdc_info.txt')
    with open(info_file) as in_file:
        subject_info = in_file.readlines()

    subject_info = [x.strip() for x in subject_info]
    subject_info = [ y.split()[0:2] + [float(z) for z in y.split()[2:]] for y in subject_info]

    
    predict_img_list = []
    predict_gt_list = []

    
    for subject in all_subjects:
        print(subject)
        subject_dir = data_dir.format(subject)
        subject_mask_original_dir = os.path.join(subject_dir, 'mask_original_2D')
        crop_2D_path = os.path.join(subject_dir, 'crop_2D')
        if not os.path.exists(crop_2D_path):
            os.makedirs(crop_2D_path)
        
        
        instants = int([x for x in subject_info if x[0] == subject][0][2])
        ed_instant = int([x for x in subject_info if x[0] == subject][0][3])
        es_instant = int([x for x in subject_info if x[0] == subject][0][4])
        slices = int([x for x in subject_info if x[0] == subject][0][5])

        used_instants_roi = [ed_instant]
        img_path_list = []
        for t in used_instants_roi:
            for s in range(int(round(slices * 0.1 + 0.001)), int(round(slices * 0.5 + 0.001))):
                s_t_mask_image_file = os.path.join(subject_mask_original_dir, 'mask_original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                img_path_list.append(s_t_mask_image_file)


    
        # Multithread
        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(determine_rectangle_roi2)
        roi_results = pool.map(function_partial, (img_path for img_path in img_path_list))
        roi_c_min = min([res[0] for res in roi_results])
        roi_c_max = max([res[1] for res in roi_results])
        roi_r_min = min([res[2] for res in roi_results])
        roi_r_max = max([res[3] for res in roi_results])
        pool.close()
        pool.join()

        # ROI size (without adding margin)
        roi_c_length = roi_c_max - roi_c_min + 1
        roi_r_length = roi_r_max - roi_r_min + 1
        roi_length = max(roi_c_length, roi_r_length)
        print('roi_length = {}'.format(roi_length) )

        written = '{0} {1} {2} {3} {4} {5}\n'.format(subject, roi_c_min, roi_c_max, roi_r_min, roi_r_max, roi_length)
        

        # The size of margin, determined by the ratio we defined above
        pixel_margin = int(round(pixel_margin_ratio * roi_length + 0.001))

        crop_c_min = ((roi_c_min + roi_c_max) // 2) - (roi_length // 2) - pixel_margin
        crop_c_max = crop_c_min + pixel_margin + roi_length - 1 + pixel_margin
        crop_r_min = ((roi_r_min + roi_r_max) // 2) - (roi_length // 2) - pixel_margin
        crop_r_max = crop_r_min + pixel_margin + roi_length - 1 + pixel_margin


        # Crop the original images
        image_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        image_load = nib.load(image_file)
        image_data = image_load.get_data()
        original_r_min = max(0, crop_r_min)
        original_r_max = min(image_data.shape[0]-1, crop_r_max)
        original_c_min = max(0, crop_c_min)
        original_c_max = min(image_data.shape[1]-1, crop_c_max)
        crop_image_data = np.zeros((roi_length + 2 * pixel_margin, roi_length + 2 * pixel_margin,
                                    image_data.shape[2], image_data.shape[3]))
        crop_image_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                        (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                        :, 
                        :] = \
            image_data[original_r_min:(original_r_max + 1), 
                       original_c_min:(original_c_max + 1), 
                       :, 
                       :]
        crop_image_data = crop_image_data[::-1, ::-1, :, :]
        crop_image_file = os.path.join(subject_dir, 'crop_{}_4d.nii.gz'.format(subject))
        nib.save(nib.Nifti1Image(crop_image_data, np.eye(4)), crop_image_file)

        # Crop the original labels
        if subject in train_subjects:
            for i in [ed_instant+1, es_instant+1]:
                label_file = os.path.join(subject_dir, '{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                label_load = nib.load(label_file)
                label_data = label_load.get_data()
                crop_label_data = np.zeros((roi_length + 2 * pixel_margin, 
                    roi_length + 2 * pixel_margin,
                    image_data.shape[2]))
                crop_label_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                        (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                        :] = \
                    label_data[original_r_min:(original_r_max + 1), 
                       original_c_min:(original_c_max + 1), 
                       :]
                crop_label_data = crop_label_data[::-1, ::-1, :]
                crop_label_file = os.path.join(subject_dir,
                    'crop_{}_frame{}_gt.nii.gz'.format(subject,str(i).zfill(2)))
                nib.save(nib.Nifti1Image(crop_label_data, np.eye(4)), crop_label_file)


    
        # Save cropped 2D images
        crop_image_data = nib.load(crop_image_file).get_data()

        max_pixel_value = crop_image_data.max()

        if max_pixel_value > 0:
            multiplier = 255.0 / max_pixel_value
        else:
            multiplier = 1.0

        print('max_pixel_value = {}, multiplier = {}'.format(max_pixel_value, multiplier) )

        for s in range(slices):
            for t in range(instants):
                s_t_image_file = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                Image.fromarray((np.rot90(crop_image_data[:, ::-1, s, t], 3) * multiplier).astype('uint8')).save(s_t_image_file)


        # Save cropped 2D labels
        if subject in train_subjects:
            for s in range(slices):
                for t in [ed_instant, es_instant]:
                    crop_label_file = os.path.join(subject_dir, 
                        'crop_{}_frame{}_gt.nii.gz'.format(subject,str(t+1).zfill(2)))
                    crop_label_data = nib.load(crop_label_file).get_data()
                    s_t_label_file = os.path.join(crop_2D_path, 'crop_2D_gt_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                    Image.fromarray((np.rot90(change_array_values(crop_label_data[:, ::-1, s]), 3) * 50).astype('uint8')).save(s_t_label_file)

    
    

    print('Done!')



if __name__ == '__main__':
    crop_according_to_roi()




