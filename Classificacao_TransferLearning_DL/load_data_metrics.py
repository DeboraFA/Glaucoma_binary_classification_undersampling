#!/usr/bin/env python
# coding: utf-8

# # Load dataset or features


import numpy as np
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(load_path):
    ### ACRIMA
    ACRIMA_data, ACRIMA_labels = load_path("D:/Glaucoma/Banco_de_dados/ACRIMA/Images")

    ## REFUGE
    REFUGE_data_train, REFUGE_labels_train = load_path("D:/Glaucoma/Banco_de_dados/REFUGE/cortes2")
    REFUGE_data_test, REFUGE_labels_test = load_path("D:/Glaucoma/Banco_de_dados/REFUGE/REFUGE-Test400/Cortes_test")

    #### RIM-ONE 
    RO_data_train, RO_labels_train = load_path("D:/Glaucoma/Banco_de_dados/RIM-ONE_DL_images/partitioned_randomly/training_set")
    RO_data_test, RO_labels_test = load_path("D:/Glaucoma/Banco_de_dados/RIM-ONE_DL_images/partitioned_randomly/test_set")


    # Refuge + Acrima + Rim-One DL
    data = np.concatenate([ACRIMA_data, REFUGE_data_train, REFUGE_data_test, RO_data_train, RO_data_test], axis = 0)
    labels_ = np.concatenate([ACRIMA_labels, REFUGE_labels_train, REFUGE_labels_test, RO_labels_train, RO_labels_test], axis = 0)

    le = LabelEncoder()
    labels = le.fit_transform(labels_)

    print("Data: ", np.shape(data))
    print("Normal: ", len(data[labels==0]))
    print("Glaucoma: ", len(data[labels==1]))


    return data, labels



# # Metrics

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score, recall_score, classification_report, confusion_matrix

def metrics(y_pred, y_true):
    acuracia = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f_score = 2*(precision*sensitivity)/(precision+sensitivity)
    
    resultado_teste = "%f %f %f %f %f %f %f %f %f" % (acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn)
    print(resultado_teste)
        
    return acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn
            


# # Balanced dataset

from imblearn.under_sampling import ClusterCentroids 
from imblearn.under_sampling import NearMiss


def data_balanced(data, labels, method):
    data_normal = data[labels==0]
    data_glaucoma = data[labels==1]
    
    if len(data_glaucoma) < len(data_normal):
        qtd_glaucoma_train = int(len(data_glaucoma))
    if len(data_glaucoma) > len(data_normal):
        qtd_glaucoma_train = int(len(data_normal))

    if method=='random':
        random.seed(10)
        qt_glaucoma_random = np.array(random.sample(range(0, len(data_glaucoma)), qtd_glaucoma_train))
        qt_normal_random = np.array(random.sample(range(0, len(data_normal)), qtd_glaucoma_train))

        data_glaucoma_train = []
        data_normal_train = []

        for i in sorted(qt_glaucoma_random):
            data_glaucoma_train.append(data_glaucoma[i])

        for j in sorted(qt_normal_random):
            data_normal_train.append(data_normal[j])

        data_train = np.concatenate([data_normal_train, data_glaucoma_train])
        labels_train = np.concatenate([np.repeat(0,len(data_normal_train)), np.repeat(1,len(data_glaucoma_train))])
        
    elif method=='nm':
        strategy = {0:qtd_glaucoma_train, 1:qtd_glaucoma_train}
        data_train, labels_train = NearMiss(version=3, sampling_strategy=strategy).fit_resample(data, labels)
        
    
    elif method=='cluster':
        strategy = {0:qtd_glaucoma_train, 1:qtd_glaucoma_train}
        data_train, labels_train = ClusterCentroids(sampling_strategy = strategy, random_state=42).fit_resample(data, labels)
        
    return data_train, labels_train



