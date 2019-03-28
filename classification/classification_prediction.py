import sys
sys.path.append('..')

import os
import numpy as np

import config

from classifiers import classifiers


def classification_prediction():
    code_path = config.code_dir

    classifier_names = ['RVA_classifier', 'HCM_classifier', 'DCM_classifier', 'MINF_classifier']
    data_class_nums = [5, 4, 3, 2]


    ###############################################
    # 5-fold cross validation on ACDC training set
    ###############################################
    print('\n5-fold cross validation on ACDC training set.')
    for k in range(4):
        print(classifier_names[k] + ' errors:')
        error_list = []
        for f in range(1, 6):
            error_list.append(classifiers(f, data_class_nums[k]))
        print(error_list)



    ##################################
    # Prediction on ACDC training set
    ##################################
    print('\nPrediction on ACDC testing set.\n')
    
    # Predictions by the 4 binary classifiers
    for k in range(4):
        classifiers(0, data_class_nums[k], classifier_names[k])
    
   
    # Assemble the results 
    test_subjects = config.acdc_test_subjects

    all_results = [[] for k in range(4)]
    for k in range(4):  
        classifier_result_file = os.path.join(code_path, 'classification', 'acdc_testing_set_prediction_{}.txt'.format(classifier_names[k]))
        with open(classifier_result_file ) as f:
            classifier_result_lines = f.readlines()
        results = [x.strip() for x in classifier_result_lines]
        results = [ [y.split()[0]] + [float(z) for z in y.split()[1:]] for y in results]
        all_results[k] = results


    rva_classifier_results = all_results[0] 
    hyp_classifier_results = all_results[1]
    dcm_classifier_results = all_results[2]   
    minf_classifier_results = all_results[3]    
    

    record_file = os.path.join(code_path, 'classification', 'acdc_testing_set_prediction.txt')
    record = open(record_file, 'w')

    
    for subject in test_subjects:
        rva_classifier_prediction = [x for x in rva_classifier_results if x[0] == subject][0][1]
        hyp_classifier_prediction = [x for x in hyp_classifier_results if x[0] == subject][0][1]
        dcm_classifier_prediction = [x for x in dcm_classifier_results if x[0] == subject][0][1]
        minf_classifier_prediction = [x for x in minf_classifier_results if x[0] == subject][0][1]
        if rva_classifier_prediction == 1:
            final_prediction = 'RV'
        elif hyp_classifier_prediction == 1:
            final_prediction = 'HCM'
        elif dcm_classifier_prediction == 1:
            final_prediction = 'DCM'
        elif minf_classifier_prediction == 1:
            final_prediction = 'MINF'
        else:
            final_prediction = 'NOR'

        written = '{} {}\n'.format(subject, final_prediction)
        record.write(written)

    record.close()




if __name__ == '__main__':
    classification_prediction()

