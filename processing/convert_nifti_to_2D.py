""" Convert the NIfTI images and ground-truth to groups of 2D PNG files """

import sys
sys.path.append('..')

import os
import numpy as np
from PIL import Image
from scipy import interpolate
import nibabel as nib

import config



def convert_nifti_to_2D():
    data_dir = config.acdc_data_dir
    code_dir = config.code_dir
    

    subjects = ['patient{}'.format(str(x).zfill(3)) for x in range(1, 151)]

    print('There are {} subjects in total'.format(len(subjects)))

    # For each case
    for subject in subjects:
        print('Processing {}'.format(subject) )

        # Define the paths
        subject_dir = data_dir.format(subject)
        subject_original_2D_dir = os.path.join(subject_dir, 'original_2D')

        if not os.path.exists(subject_original_2D_dir):
            os.makedirs(subject_original_2D_dir)

        sa_zip_file = os.path.join(subject_dir, '{}_4d.nii.gz'.format(subject))
        
        # If the short-axis image file exists, read the data and perform the conversion
        if os.path.isfile(sa_zip_file):
            img = nib.load(sa_zip_file)
            data = img.get_data()
            data_np = np.array(data)

            max_pixel_value = data_np.max()

            if max_pixel_value > 0:
                multiplier = 255.0 / max_pixel_value
            else:
                multiplier = 1.0

            print('max_pixel_value = {},  multiplier = {}'.format(max_pixel_value, multiplier))

            rows = data.shape[0]
            columns = data.shape[1]
            slices = data.shape[2]
            times = data.shape[3]

            for t in range(times):
                for s in range(slices):
                    s_t_image_file = os.path.join(subject_original_2D_dir, 'original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                    Image.fromarray((np.rot90(data[:, ::-1, s, t], 1) * multiplier).astype('uint8')).save(s_t_image_file)

        else:
            print('There is no SA image file for {}'.format(subject))


        



if __name__ == '__main__':
    convert_nifti_to_2D()

    




