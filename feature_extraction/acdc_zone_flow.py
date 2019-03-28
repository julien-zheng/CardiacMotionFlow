import numpy as np
import os

from helpers import (
    mask_barycenter2,
    masked_flow_transform2,
    flow_by_zone3,
    enlarge_mask4
)
from image2 import load_img2
import config



def acdc_zone_flow():


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


    num_zone = 6
    zfill_num = 2
    img_size = config.apparentflow_net_input_img_size
    shape = (img_size, img_size ,2)
    



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

        for slice_idx in range(start_slice, end_slice):
            print('slice #{}'.format(slice_idx))

            # Get the mask
            mask_file = folder + 'predict_lvrv2_{}_{}.png'.format(str(slice_idx).zfill(zfill_num), str(ed_instant).zfill(zfill_num))
            mask = load_img2(mask_file, grayscale=True, 
                         target_size=(shape[0], 
                         shape[1]),
                         pad_to_square=True, resize_mode='nearest')
            mask = np.reshape(np.array(mask)/50.0, (1, shape[0], shape[1], 1))
            mask = enlarge_mask4(mask, width=1, enlarge_value=2.0, neighbor_values=[1.0])


            # Get the flow
            flow = np.zeros((1, shape[0], shape[1], 0))
            for idx in idx_range:
                flow_file = folder + 'flow2_{}_{}.npy'.format(str(slice_idx).zfill(zfill_num), str(idx).zfill(zfill_num))
                if idx != idx_range[0] and os.path.isfile(flow_file):
                    flow_idx = np.reshape(np.load(flow_file), (1, shape[0], shape[1], shape[2]))
                else:
                    flow_idx = np.zeros((1, shape[0], shape[1], shape[2]))
                flow = np.concatenate((flow, flow_idx), axis=-1)


            # Compute the barycenter coordinates
            lv_mask = np.where(np.logical_or(\
                                  np.equal(mask, 2.0 * np.ones_like(mask)), 
                                  np.equal(mask, 1.0 * np.ones_like(mask)) ),
                                  np.ones_like(mask), np.zeros_like(mask))
            rv_mask = np.where(np.equal(mask, 3.0 * np.ones_like(mask)), 
                               np.ones_like(mask), np.zeros_like(mask))
            barycenters = mask_barycenter2(flow, mask, mask_value=1.0)
            rv_barycenters = mask_barycenter2(flow, mask, mask_value=3.0)


            # Transform the flow
            transformed_flow, angles, distance_flows, norms, boundary_pixels = \
                masked_flow_transform2(flow, mask, barycenters, lvm_value = 2.0, lvc_value = 1.0)

            # Average the transformed flow by zone
            zone_avg_flow, zone_std_original_flow, \
            zone_avg_inner_border_normalized_flow, zone_avg_outer_border_normalized_flow, \
            zone_avg_myo_thickness_flow, zone_map = \
                flow_by_zone3(transformed_flow, flow, angles, distance_flows, norms, boundary_pixels, num_zone, start_random = False, barycenters = barycenters, rv_barycenters = rv_barycenters)

            # Normalize the flow
            zone_avg_inner_border_normalized_flow *= normalize_term
            zone_avg_myo_thickness_flow *= normalize_term

            np.save(folder + 'radius_flow_{}.npy'.format(str(slice_idx).zfill(zfill_num)), zone_avg_inner_border_normalized_flow)
            np.save(folder + 'thickness_flow_{}.npy'.format(str(slice_idx).zfill(zfill_num)), zone_avg_myo_thickness_flow)



if __name__ == '__main__':
    acdc_zone_flow()










        
