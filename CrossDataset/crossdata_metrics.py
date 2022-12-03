#!/usr/bin/env python
# coding: utf-8

# # Balanced dataset
import numpy as np
from imblearn.under_sampling import ClusterCentroids 
from imblearn.under_sampling import NearMiss
import random

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
            



from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
def train_crossdataset(X_tr, y_tr, X_test, Y_test, balanced, models, method):
    
    if balanced=='None':
        X_train = X_tr
        Y_train = y_tr

    elif balanced=='random':
        X_train, Y_train = data_balanced(X_tr, y_tr, 'random')
  
    elif balanced=='nm':
        X_train, Y_train = data_balanced(X_tr, y_tr, 'nm')

    elif balanced=='cluster':
        X_train, Y_train = data_balanced(X_tr, y_tr, 'cluster')

    
    if method =='scaler':
        scaler = StandardScaler()
        scaler.fit(X_train)
        train_img = scaler.transform(X_train) 
        test_img = scaler.transform(X_test) 
    else:
        train_img = X_train
        test_img = X_test
    
    for name, model in models: 
        modelo = model.fit(train_img, Y_train)
        y_pred = modelo.predict(test_img)
        y_true = Y_test
        acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn = metrics(y_pred, y_true)
            

    return
