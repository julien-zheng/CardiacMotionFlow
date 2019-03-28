# Explainable cardiac pathology classification on cine MRI with motion characterization by semi-supervised learning of apparent flow

This is an implementation of the models in the following paper which is submitted to the **Medical Image Analysis** journal:

	Explainable cardiac pathology classification on cine MRI with motion characterization by semi-supervised learning of apparent flow
	Qiao Zheng, Hervé Delingette, Nicholas Ayache
	


**In case you find the code useful, please consider giving appropriate credit to it by citing the paper above.**

```
@ARTICLE{Qiao:MEDIA:2018,
	Author = {Zheng, Q and Delingette, H and Ayache, N},
	Journal = {?},
	Title = {Explainable cardiac pathology classification on cine MRI with motion characterization by semi-supervised learning of apparent flow},
	Volume = {?},
	Pages = {?-?},
	Year = {2018}
}

```

```
DOI: ?
```

## Requirements
The code should work with both Python 2 and Python 3, as the compatibility with both has been taken into consideration. It depends on some external libraries which should be installed, including:
- tensorflow
- keras 
- numpy and scipy
- math
- PIL
- cv2
- nibabel
- itertools
- copy
- six
- threading
- warnings
- multiprocessing
- functools
- h5py
- sklearn
- urllib

On the other hand, **to apply this package on any data, the images should be named, formatted, and arranged accordingly.**. Please refer to Section VIII for more details.


## Usage
The steps of the cardiac pathology classification method presented in the paper are described below, along with the corresponding files. **First, modify the `code_dir` and `acdc_data_dir` in *config.py* to the corresponding path of the code root directory and the ACDC data root directory.** This is the very first step in the use of the software as these paths are necessary for the other scripts. The default values of the other variables in *config.py* are those used by the paper. Then, read the following sections according to your application scenario:
- If you want to train and test the model yourself on the ACDC dataset as we have done, read from Section I to Section VI.
- If you are only interested in applying the pretrained model on the ACDC testing set, you may mainly focus on the sections I, II, III.2, IV.2, V and VI.
- If you want to train and/or test the model using another dataset instead of the ACDC dataset, useful details are provided in Section VIII.

Depending on the version of this package, it may or may not contain the trained weights of the networks (the .h5 files for ROI-net, LVRV-net and ApparentFlow-net). For instance, due to the constraints on the size of files, the version on GitHub does not contain the trained weights. **If the trained weights are not included in the package, they can be downloaded by running**:
```
python download_weights.py
```

### I. Preprocessing of ACDC Data
In this section, the preprocessing of ACDC data is presented step by step. However, it is also possible to preprocess other datasets accordingly and then use them to train or test our model (more details are presented in Section VIII).

#### I.1 Download the data
Download the data to a directory (and adapt `acdc_data_dir` in *config.py* accordingly). Make sure that in this directory, the data of each of the 150 cases, including 100 in the training set and 50 in the testing set, is in a sub-directory named "patientxxx" (e.g. "patient001", "patient068", "patient124") following the convention of ACDC.

#### I.2 Save as 2D images
```
python processing/convert_nifti_to_2D.py
```
To accelerate the reading of the image of a short-axis slice at an instant, the short-axis images, as well as their ground-truth segmentation (if exists), are converted to and saved as PNG format. The following file is for this step:
- processing/convert_nifti_to_2D.py

#### I.3 Basic information
```
python processing/acdc_info.py
```
Save the useful basic information (e.g. number of slices and frames) of the subjects to a file. The following file is for this step:
- processing/acdc_info.py

The resulting information is saved to *acdc_info/acdc_info.txt*.

#### I.4 Determination of key slices based on ground-truth segmentation
```
python processing/acdc_gt_base.py 
```
Some key slices (e.g. the first slice below the base and the last slice above the apex) are determined based on ground-truth segmentation using:
- processing/acdc_gt_base.py 

The resulting information is saved to *acdc_info/acdc_gt_base.txt*. This information is used in the training process only.


### II. Region of Interest (ROI) Determination Using ROI-net
ROI-net, the network for ROI determination as presented and trained in our previous paper "3D Consistent & Robust Segmentation of Cardiac Images by Deep Learning with Spatial Propagation" is applied on the ACDC data.

#### II.1 Prediction
```
python ROI/predict_roi_net.py
```
- ROI/predict_roi_net.py: the main file to launch the prediction
- ROI/data_roi_predict.py: a function to generate lists of files for prediction
- ROI/module_roi_net.py: definition of the network
- ROI/model_roi_net_epoch050.h5: the weights of the trained network

#### II.2 ROI cropping
```
python ROI/crop_according_to_roi.py
```
- ROI/crop_according_to_roi.py: crop and save the determined ROIs



### III. Apparent Flow Generation Using ApparentFlow-net
ApparentFlow-net is trained on the whole ACDC training set for prediction on the ACDC tesing set (this version is called "fold 0" in this project); it is also trained for a 5-fold cross-validation on the ACDC training set (called fold 1, 2, 3, 4 and 5 respectively). 
#### III.1 Training
For fold `f` = 0, 1, 2, 3, 4 and 5, run:
```
python flow/train_apparentflow_net.py f
```
- flow/train_apparentflow_net.py: the main file to launch the training
- flow/data_apparentflow.py: a function to generate lists of files for training and prediction
- flow/module_apparentflow_net.py: the file that defines ApparentFlow-net
- flow/model_apparentflow_net_fold`f`_epoch050.h5: the weights of the trained network for fold `f` = 0, 1, 2, 3, 4 and 5

#### III.2 Prediction
For fold `f` = 0, 1, 2, 3, 4 and 5, run:
```
python flow/predict_apparentflow_net.py f
```
- flow/predict_lvrv_net.py: the main file to launch the prediction



### IV. Cardiac Segmentation Using LVRV-net
LVRV-net, the network for cardiac segmentation as presented and trained in our previous paper "3D Consistent & Robust Segmentation of Cardiac Images by Deep Learning with Spatial Propagation" is finetuned and then applied on the ACDC data.

#### IV.1 Finetuning
For fold `f` = 0, 1, 2, 3, 4 and 5, run:
```
python segmentation/finetune_lvrv_net.py f
```
- segmentation/finetune_lvrv_net.py: the main file to launch the finetuning
- segmentation/data_lvrv_segmentation_propagation_acdc.py: a function to generate lists of files for finetuning and prediction
- segmentation/module_lvrv_net.py: the file that defines LVRV-net
- segmentation/model_lvrv_net_epoch080.h5: the weights of the network trained on UK Biobank
- segmentation/model_lvrv_net_finetune_fold`f`_epoch1000.h5: the weights of the network trained on UK Biobank and then finetuned on ACDC for fold `f` = 0, 1, 2, 3, 4 and 5

#### IV.2 Prediction
For fold `f` = 0, 1, 2, 3, 4 and 5, run:
```
python segmentation/prediction_lvrv_net.py f
```
- segmentation/predict_lvrv_net.py: the main file to launch the prediction



### V. Feature Extraction
#### V.1 Extraction of Shape-Related Features
```
python feature_extraction/acdc_base.py
python feature_extraction/acdc_pixel_size.py
python feature_extraction/acdc_thickness.py
python feature_extraction/acdc_volume.py
```
- feature_extraction/acdc_base.py: determination of the key slices (e.g. the first slice below the base and the last slice above the apex) based on the segmentation predicted by LVRV-net. The resulting information is saved to *acdc_info/acdc_base.txt*
- feature_extraction/acdc_pixel_size.py: extract the pixel size of the images. The resulting information is saved to *acdc_info/acdc_pixel_size.txt*
- feature_extraction/acdc_thickness.py: extract the thickness of myocardium based on the segmentation predicted by LVRV-net. The resulting information is saved to *acdc_info/acdc_thickness.txt*
- feature_extraction/acdc_volume.py: extract the cardiac volumes based on the segmentation predicted by LVRV-net. The resulting information is saved to *acdc_info/acdc_volume.txt*

#### V.2 Extraction of Motion-Characteristic Features
```
python feature_extraction/acdc_zone_flow.py
python feature_extraction/acdc_motion_index.py
```
- feature_extraction/acdc_zone_flow.py: extract the time series characterizing cardiac segmental motion based on the apparent flow predicted by ApparentFlow-net. The resulting time series are saved to the corresponding sub-directories in `acdc_data_dir`
- feature_extraction/acdc_motion_index.py: extract the features characterizing cardiac motion based on the apparent flow predicted by ApparentFlow-net. The resulting information is saved to *acdc_info/acdc_motion_index.txt*


### VI. Classification
```
python classification/classification_prediction.py
```
- classification/classification_prediction.py: the main file to launch the classification
- classification/data_classification.py: a function to generate lists of files for classification
- classification/classifiers.py: the file that defines the binary classifiers and their respective training and prediction processes
- classification/trained_model_`XXX`_classifier.joblib: the 4 binary classifiers trained on the ACDC training set, for `XXX` = RVA, HCM, DCM and MINF
- classification/acdc_testing_set_prediction_`XXX`_classifier.txt: the prediction on the ACDC testing set by the 4 binary classifiers trained on the ACDC training set, for `XXX` = RVA, HCM, DCM and MINF
- classification/acdc_testing_set_prediction.txt: the prediction on the ACDC testing set by the overall classification model (which consists of the 4 binary classifiers) trained on the ACDC training set



### VII. Auxiliary functions
Just for information, the auxiliary functions are defined in the following files:
- helpers.py
- image2.py

In particular, image2.py is used for real-time data augmentation. It is adapted from the code in a file of the Keras project (https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py).



### VIII. Using Other Datasets
Instead of using the ACDC data, it is also possible to preprocess another dataset accordingly and then use it to train and/or test our model, as long as the following conventions on name and format of the preprocessed short-axis images are met.

In the data directory, the path of a 2D image, which is identified by its case name string `C` (e.g. 'patient007'), the two-digit slice index `S` (e.g. 02) in the stack, and the two-digit instant index `I` (e.g. 06) in the temporal sequence, should be the following:
- *`'{C}/original_2D/original_2D_{S}_{I}.png'`* (e.g. *'patient007/original_2D/original_2D_02_06.png'*)

The corresponding ground-truth should have the path:
- *`'{C}/original_2D/original_gt_2D_{S}_{I}.png'`* (e.g. *'patient007/original_2D/original_gt_2D_02_16.png'*)

**Please note that the two-digit slice index `S` in the stack should be arranged to increment slice by slice from the base to the apex. This is essential as it makes sure that the propagation is performed in the correct base-to-apex order.**













