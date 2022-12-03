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





# # Metrics

from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score, recall_score, classification_report, confusion_matrix

def metrics(y_pred, y_true):
    acuracia = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f_score = 2*(precision*sensitivity)/(precision+sensitivity)
    
#     resultado_teste = "%f %f %f %f %f %f %f %f %f" % (acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn)
#     print(resultado_teste)
        
    return acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn
            



# # Cross Validation
import pickle
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

def cv_train(models, data_comp, labels_comp, n_splits, balanced, method):
    
   
    if balanced=='None':
        data = data_comp
        labels = labels_comp
    elif balanced=='random':
        data, labels = data_balanced(data_comp, labels_comp, 'random')
    elif balanced=='nm':
        data, labels = data_balanced(data_comp, labels_comp, 'nm')
    elif balanced=='cluster':
        data, labels = data_balanced(data_comp, labels_comp, 'cluster')

    
    P = []
    A = []
    E = []
    S = []
    Fs = []
    TP = []
    TN = []
    FP = []
    FN = []
    
#     kfold = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=10)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10)
    
    for name, model in models: 
        
        for train_ix, test_ix in kfold.split(data, labels):

            X_train, X_test = data[train_ix], data[test_ix]
            Y_train, Y_test = labels[train_ix], labels[test_ix]
           
            
            
            if method =='scaler':
                scaler = preprocessing.StandardScaler()
                scaler.fit(X_train)
                train_img = scaler.transform(X_train) 
                test_img = scaler.transform(X_test) 
            else: 
                train_img = X_train
                test_img = X_test


            ################# Validação k-fold ##################
            
            modelo = model.fit(train_img, Y_train)
            y_pred = modelo.predict(test_img)
            y_true = Y_test
            
#             pickle.dump(modelo, open('models/model_' + name + '_' + balanced +'.sav', 'wb'))

            acuracia, precision, sensitivity, specificity, f_score, tp, tn, fp, fn = metrics(y_pred, y_true)
            
            P.append(precision)
            A.append(acuracia)
            E.append(specificity)
            S.append(sensitivity)     
            Fs.append(f_score)
            TP.append(tp)
            TN.append(tn)
            FP.append(fp)
            FN.append(fn)

            resultado_teste = "%s: %f %f %f %f %f %f %f %f %f" % (name,acuracia,precision,sensitivity,specificity,f_score,tp,tn,fp,fn)
            print(resultado_teste)
        
    return P, A, S, E, Fs, TP, TN, FP, FN


# # Summary results


def metrics_classify(init, P, A, S, E, Fs, TP, TN, FP, FN):
  
    P_class = []
    A_class = []
    S_class = []
    E_class = []
    f_class = [] 
    TP_class = []
    TN_class = []    
    FP_class = []
    FN_class = []
  
  
    for i in np.arange(init*10, (init+1)*10, 1):
        P_class.append(P[i])
        A_class.append(A[i])
        S_class.append(S[i])
        E_class.append(E[i])
        f_class.append(Fs[i])
        TP_class.append(TP[i])
        TN_class.append(TN[i])
        FP_class.append(FP[i])
        FN_class.append(FN[i])

    return P_class, A_class, S_class, E_class, f_class, TP_class, TN_class, FP_class, FN_class
  
def result_average(P, A, S, E, fb, TP, TN, FP, FN):
    acc_media = np.mean(A)
    acc_dp = np.std(A)
    prec_media = np.mean(P)
    prec_dp = np.std(P)
    sens_media = np.mean(S)
    sens_dp = np.std(S)
    esp_media = np.mean(E)
    esp_dp = np.std(E)
    f1_media = np.mean(fb)
    f1_dp = np.std(fb)

    acc_tp = np.mean(TP)
    acc_tn = np.mean(TN)
    acc_fp = np.mean(FP)
    acc_fn = np.mean(FN)

    return prec_media*100, prec_dp*100, acc_media*100, acc_dp*100, sens_media*100, sens_dp*100, esp_media*100, esp_dp*100, f1_media*100, f1_dp*100, acc_tp, acc_tn, acc_fp, acc_fn
  
def results(P, A, S, E, Fs, TP, TN, FP, FN):
    P_xgb, A_xgb, S_xgb, E_xgb, f1_xgb, TP_xgb, TN_xgb, FP_xgb, FN_xgb = metrics_classify(0, P, A, S, E, Fs, TP, TN, FP, FN)
    P_svm, A_svm, S_svm, E_svm, f1_svm, TP_svm, TN_svm, FP_svm, FN_svm = metrics_classify(1, P, A, S, E, Fs, TP, TN, FP, FN)
    P_mlp, A_mlp, S_mlp, E_mlp, f1_mlp, TP_mlp, TN_mlp, FP_mlp, FN_mlp = metrics_classify(2, P, A, S, E, Fs, TP, TN, FP, FN)
    P_vot, A_vot, S_vot, E_vot, f1_vot, TP_vot, TN_vot, FP_vot, FN_vot = metrics_classify(3, P, A, S, E, Fs, TP, TN, FP, FN)

    prec_media_xgb, prec_dp_xgb, acc_media_xgb, acc_dp_xgb, sens_media_xgb, sens_dp_xgb, esp_media_xgb, esp_dp_xgb, f1_media_xgb, f1_dp_xgb, TP_media_xgb, TN_media_xgb, FP_media_xgb, FN_media_xgb = result_average(P_xgb, A_xgb, S_xgb, E_xgb, f1_xgb, TP_xgb, TN_xgb, FP_xgb, FN_xgb)
    prec_media_svm, prec_dp_svm, acc_media_svm, acc_dp_svm, sens_media_svm, sens_dp_svm, esp_media_svm, esp_dp_svm, f1_media_svm, f1_dp_svm, TP_media_svm, TN_media_svm, FP_media_svm, FN_media_svm = result_average(P_svm, A_svm, S_svm, E_svm, f1_svm, TP_svm, TN_svm, FP_svm, FN_svm)
    prec_media_mlp, prec_dp_mlp, acc_media_mlp, acc_dp_mlp, sens_media_mlp, sens_dp_mlp, esp_media_mlp, esp_dp_mlp, f1_media_mlp, f1_dp_mlp, TP_media_mlp, TN_media_mlp, FP_media_mlp, FN_media_mlp = result_average(P_mlp, A_mlp, S_mlp, E_mlp, f1_mlp, TP_mlp, TN_mlp, FP_mlp, FN_mlp)
    prec_media_vot, prec_dp_vot, acc_media_vot, acc_dp_vot, sens_media_vot, sens_dp_vot, esp_media_vot, esp_dp_vot, f1_media_vot, f1_dp_vot, TP_media_vot, TN_media_vot, FP_media_vot, FN_media_vot = result_average(P_vot, A_vot, S_vot, E_vot, f1_vot, TP_vot, TN_vot, FP_vot, FN_vot)


    res_test_xgb = "%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f & %f & %f & %f" % (acc_media_xgb, acc_dp_xgb, prec_media_xgb, prec_dp_xgb,  sens_media_xgb, sens_dp_xgb, esp_media_xgb, esp_dp_xgb, f1_media_xgb, f1_dp_xgb, TP_media_xgb, TN_media_xgb, FP_media_xgb, FN_media_xgb)
    res_test_svm = "%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f & %f & %f & %f" % (acc_media_svm, acc_dp_svm, prec_media_svm, prec_dp_svm,  sens_media_svm, sens_dp_svm, esp_media_svm, esp_dp_svm, f1_media_svm, f1_dp_svm, TP_media_svm, TN_media_svm, FP_media_svm, FN_media_svm)
    res_test_mlp = "%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f & %f & %f & %f" % (acc_media_mlp, acc_dp_mlp, prec_media_mlp, prec_dp_mlp, sens_media_mlp, sens_dp_mlp, esp_media_mlp, esp_dp_mlp, f1_media_mlp, f1_dp_mlp, TP_media_mlp, TN_media_mlp, FP_media_mlp, FN_media_mlp)
    res_test_vot = "%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f & %f & %f & %f" % (acc_media_vot, acc_dp_vot, prec_media_vot, prec_dp_vot, sens_media_vot, sens_dp_vot, esp_media_vot, esp_dp_vot, f1_media_vot, f1_dp_vot, TP_media_vot, TN_media_vot, FP_media_vot, FN_media_vot)

    print(res_test_xgb)
    print(res_test_svm)
    print(res_test_mlp)
    print(res_test_vot)

    
    # Acurácia Precisão Sensibilidade Especificidade Fscore True_positive True_negative False_positive False_negative
    
    return 


