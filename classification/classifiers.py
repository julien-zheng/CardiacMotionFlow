""" The file of the binary classifiers """

import sys
sys.path.append('..')
sys.path.append('.')

import os

from sklearn import svm, linear_model, ensemble, neural_network
from sklearn.externals import joblib


from data_classification import data_classification


import config



def classifiers(fold, data_class_num, classifier_name=None):
    code_path = config.code_dir

    #print('fold = {}'.format(fold))
    if fold == 0:
        mode_train = 'all'
        mode_val = 'predict'
    elif fold in range(1,6):
        mode_train = 'train'
        mode_val = 'val'
    else:
        print('Incorrect fold')


    model_type = 'lr'

    normalization = True
    
    train_list_subject_idx, train_list_lv_volume, train_list_rv_volume, train_list_lv_ratio, train_list_rv_ratio, train_list_lvmrv_ratio, train_list_lvmlv_ratio, train_list_lvmlv_mass, train_list_thickness, train_list_thickness_diff, train_list_asyn_radius, train_list_asyn_thickness, train_list_gt = data_classification(mode = mode_train, fold=fold, data_class_num=data_class_num, normalization=normalization)
    train_sample = len(train_list_subject_idx)
    
    val_list_subject_idx, val_list_lv_volume, val_list_rv_volume, val_list_lv_ratio, val_list_rv_ratio, val_list_lvmrv_ratio, val_list_lvmlv_ratio, val_list_lvmlv_mass, val_list_thickness, val_list_thickness_diff, val_list_asyn_radius, val_list_asyn_thickness, val_list_gt = data_classification(mode = mode_val, fold=fold, data_class_num=data_class_num, normalization=normalization)
    val_sample = len(val_list_subject_idx)



    train_id = [0 for x in range(train_sample)]
    train_list = [[] for x in range(train_sample)]
    train_gt = [0. for x in range(train_sample)]

    val_id = [0 for x in range(val_sample)]
    val_list = [[] for x in range(val_sample)]
    val_gt = [0. for x in range(val_sample)]  



    
    if data_class_num == 5:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_rv_volume[i])
            train_list[i].append(train_list_rv_ratio[i])
            train_list[i].append(train_list_lvmrv_ratio[i])
            train_gt[i] = float(train_list_gt[i] == 4) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_rv_volume[i])
            val_list[i].append(val_list_rv_ratio[i])
            val_list[i].append(val_list_lvmrv_ratio[i])
            val_gt[i] = float(val_list_gt[i] == 4) 
    elif data_class_num == 4:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_lv_ratio[i])
            train_list[i].append(train_list_lvmlv_ratio[i])
            train_list[i].append(train_list_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 1) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_lv_ratio[i])
            val_list[i].append(val_list_lvmlv_ratio[i])
            val_list[i].append(val_list_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 1) 
    elif data_class_num == 3:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_lv_volume[i])
            train_list[i].append(train_list_asyn_radius[i])
            train_list[i].append(train_list_asyn_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 0) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_lv_volume[i])
            val_list[i].append(val_list_asyn_radius[i])
            val_list[i].append(val_list_asyn_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 0) 
    elif data_class_num == 2:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_lv_ratio[i])
            train_gt[i] = float(train_list_gt[i] == 2) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_lv_ratio[i])
            val_gt[i] = float(val_list_gt[i] == 2)
    else:
        print('Incorrect data_class_num')
    


    '''
    if data_class_num == 5:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_rv_volume[i])
            train_list[i].append(train_list_rv_ratio[i])
            train_list[i].append(train_list_lvmrv_ratio[i])
            train_list[i].append(train_list_lv_ratio[i])
            train_list[i].append(train_list_lvmlv_ratio[i])
            train_list[i].append(train_list_thickness[i])
            train_list[i].append(train_list_lv_volume[i])
            train_list[i].append(train_list_asyn_radius[i])
            train_list[i].append(train_list_asyn_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 4) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_rv_volume[i])
            val_list[i].append(val_list_rv_ratio[i])
            val_list[i].append(val_list_lvmrv_ratio[i])
            val_list[i].append(val_list_lv_ratio[i])
            val_list[i].append(val_list_lvmlv_ratio[i])
            val_list[i].append(val_list_thickness[i])
            val_list[i].append(val_list_lv_volume[i])
            val_list[i].append(val_list_asyn_radius[i])
            val_list[i].append(val_list_asyn_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 4) 
    elif data_class_num == 4:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_rv_volume[i])
            train_list[i].append(train_list_rv_ratio[i])
            train_list[i].append(train_list_lvmrv_ratio[i])
            train_list[i].append(train_list_lv_ratio[i])
            train_list[i].append(train_list_lvmlv_ratio[i])
            train_list[i].append(train_list_thickness[i])
            train_list[i].append(train_list_lv_volume[i])
            train_list[i].append(train_list_asyn_radius[i])
            train_list[i].append(train_list_asyn_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 1) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_rv_volume[i])
            val_list[i].append(val_list_rv_ratio[i])
            val_list[i].append(val_list_lvmrv_ratio[i])
            val_list[i].append(val_list_lv_ratio[i])
            val_list[i].append(val_list_lvmlv_ratio[i])
            val_list[i].append(val_list_thickness[i])
            val_list[i].append(val_list_lv_volume[i])
            val_list[i].append(val_list_asyn_radius[i])
            val_list[i].append(val_list_asyn_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 1) 
    elif data_class_num == 3:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_rv_volume[i])
            train_list[i].append(train_list_rv_ratio[i])
            train_list[i].append(train_list_lvmrv_ratio[i])
            train_list[i].append(train_list_lv_ratio[i])
            train_list[i].append(train_list_lvmlv_ratio[i])
            train_list[i].append(train_list_thickness[i])
            train_list[i].append(train_list_lv_volume[i])
            train_list[i].append(train_list_asyn_radius[i])
            train_list[i].append(train_list_asyn_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 0) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_rv_volume[i])
            val_list[i].append(val_list_rv_ratio[i])
            val_list[i].append(val_list_lvmrv_ratio[i])
            val_list[i].append(val_list_lv_ratio[i])
            val_list[i].append(val_list_lvmlv_ratio[i])
            val_list[i].append(val_list_thickness[i])
            val_list[i].append(val_list_lv_volume[i])
            val_list[i].append(val_list_asyn_radius[i])
            val_list[i].append(val_list_asyn_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 0) 
    elif data_class_num == 2:
        for i in range(train_sample):
            train_id[i] = train_list_subject_idx[i]
            train_list[i].append(train_list_rv_volume[i])
            train_list[i].append(train_list_rv_ratio[i])
            train_list[i].append(train_list_lvmrv_ratio[i])
            train_list[i].append(train_list_lv_ratio[i])
            train_list[i].append(train_list_lvmlv_ratio[i])
            train_list[i].append(train_list_thickness[i])
            train_list[i].append(train_list_lv_volume[i])
            train_list[i].append(train_list_asyn_radius[i])
            train_list[i].append(train_list_asyn_thickness[i])
            train_gt[i] = float(train_list_gt[i] == 2) 
        for i in range(val_sample):
            val_id[i] = val_list_subject_idx[i]
            val_list[i].append(val_list_rv_volume[i])
            val_list[i].append(val_list_rv_ratio[i])
            val_list[i].append(val_list_lvmrv_ratio[i])
            val_list[i].append(val_list_lv_ratio[i])
            val_list[i].append(val_list_lvmlv_ratio[i])
            val_list[i].append(val_list_thickness[i])
            val_list[i].append(val_list_lv_volume[i])
            val_list[i].append(val_list_asyn_radius[i])
            val_list[i].append(val_list_asyn_thickness[i])
            val_gt[i] = float(val_list_gt[i] == 2)
    else:
        print('Incorrect data_class_num')
    '''
    
    

    if model_type == 'lr':
        clf = linear_model.LogisticRegression(C=50., solver='liblinear', random_state=0)
        clf.fit(train_list, train_gt)
        #print(clf.fit(train_list, train_gt))

        
        prediction = clf.predict(val_list)
        prediction_proba = clf.predict_proba(val_list)

        if fold in range(1,6):
            error = [x for i,x in enumerate(val_id) if prediction[i] != val_gt[i]]
            return error
        
        elif fold == 0:
            # Save the classifier model
            joblib.dump(clf, os.path.join(code_path, 'classification', 'trained_model_{}.joblib'.format(classifier_name)))

            #print(clf.coef_)
            #print(clf.intercept_)
        
            # Save the predicted results
            record_file = os.path.join(code_path, 'classification', 'acdc_testing_set_prediction_{}.txt'.format(classifier_name))
            record = open(record_file, 'w')
            for i in range(len(val_id)):
                subject = 'patient{}'.format(str(val_id[i]).zfill(3))
                prediction_result = prediction[i]
                proba = prediction_proba[i][1]
                written = '{} {} {}\n'.format(subject, prediction_result, proba)
                record.write(written)
            
            record.close()
            return





