import numpy as np
import os

import config



def acdc_motion_index():


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
    

    motion_info = open(os.path.join(code_dir, 'acdc_info', 'acdc_motion_index.txt'), 'w')

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


        slice_range = range(base_slice + int(round((apex_slice + 1 - base_slice)*excluded_slice_ratio*0.5)), apex_slice + 1 - int(round((apex_slice + 1 - base_slice)*excluded_slice_ratio*1)))
        


        all_radius_flow = np.zeros((1, 0, seq_instants))
        
        asyn_thickness = 0.0
        thickness_diff = -10.0
        for slice_idx in slice_range:
            zone_avg_inner_border_normalized_flow = np.load(folder + 'radius_flow_{}.npy'.format(str(slice_idx).zfill(zfill_num)))
            zone_avg_myo_thickness_flow = np.load(folder + 'thickness_flow_{}.npy'.format(str(slice_idx).zfill(zfill_num)))

            all_radius_flow = np.concatenate((all_radius_flow, zone_avg_inner_border_normalized_flow), axis=1)
            
            scale_thickness = zone_avg_myo_thickness_flow[0, :, 0].min()
            slice_asyn_thickness = (zone_avg_myo_thickness_flow[0, :, :].max(axis=0) - zone_avg_myo_thickness_flow[0, :, :].min(axis=0)) / scale_thickness
            asyn_thickness = max(asyn_thickness, slice_asyn_thickness.max())

            segment_thickness_diff = zone_avg_myo_thickness_flow[0, :, :].max(axis=1) - zone_avg_myo_thickness_flow[0, :, 0]
            thickness_diff = max(thickness_diff, segment_thickness_diff.max())

            

            
            

        for i in range(all_radius_flow.shape[1]):
            all_radius_flow[0, i, :] = (all_radius_flow[0, i, 0] - all_radius_flow[0, i, :]) /  all_radius_flow[0, i, 0] 
        
        asyn_radius = (all_radius_flow[0, :, :].max(axis=0) - all_radius_flow[0, :, :].min(axis=0)).max()
        


        written = '{} {} {} {}\n'.format(subject, asyn_radius, asyn_thickness, thickness_diff)
        motion_info.write(written)


    motion_info.close()
    



if __name__ == '__main__':
    acdc_motion_index()










        
