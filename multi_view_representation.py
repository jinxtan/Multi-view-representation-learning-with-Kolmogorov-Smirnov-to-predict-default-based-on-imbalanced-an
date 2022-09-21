# -*- coding: utf-8 -*-
"""
Created on 2021/5/11 19:34
@author: Jinxtan
@Institution: Fudan University, Shanghai, China
@email: tanyd20@fudan.edu.cn
@File: multi_view_representation.py
"""
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesClassifier
from models import resnet_finance, resnet_newfinance
from utils import lr_schedule
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import heapq
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def multi_view_AMV(train_x, test_x, train_y, test_y, val_x, val_y, header_name, class_n):
    # pca = PCA(n_components=5)
    # newX = pca.fit_transform(train_x)

    model = ExtraTreesClassifier()
    model.fit(train_x, train_y)
    # print(classification_report(test_y, model.predict(test_x)))
    d = model.feature_importances_
    tmp = zip(range(len(d)), d)
    large5 = heapq.nlargest(60, tmp, key = lambda x:x[1])
    index_new = []
    for i in range(len(large5)):
        index_new.append(large5[i][0])

    train_x_new = pd.DataFrame()
    test_x_new = pd.DataFrame()
    val_x_new = pd.DataFrame()
    for i in index_new:
        train_x_new[header_name[i]] = train_x[:,i]
        test_x_new[header_name[i]] = test_x[:, i]
        val_x_new[header_name[i]] = val_x[:, i]
    train_x_new['loan_status'] = train_y
    corr = train_x_new.corr(method='spearman')
    corr.to_csv('corr_result.csv')
    corr = corr.drop('loan_status', axis=0)

    p_li = []
    n_li = []
    p_lf = []
    n_lf = []

    for i in range(len(corr)):
        if corr['loan_status'][i]>0:
            if class_n[i] <= np.median(class_n):
                p_li.append(corr.index[i])
            else:
                p_lf.append(corr.index[i])
        else:
            if class_n[i] <= np.median(class_n):
                n_li.append(corr.index[i])
            else:
                n_lf.append(corr.index[i])
    data_pi = train_x_new[p_li]
    data_ni = train_x_new[n_li]
    data_pf = train_x_new[p_lf]
    data_nf = train_x_new[n_lf]
    pd.Series([p_li,n_li,p_lf,n_lf]).to_csv('header.csv')
    data_pi_t = test_x_new[p_li]
    data_ni_t = test_x_new[n_li]
    data_pf_t = test_x_new[p_lf]
    data_nf_t = test_x_new[n_lf]

    data_pi_v = val_x_new[p_li]
    data_ni_v = val_x_new[n_li]
    data_pf_v = val_x_new[p_lf]
    data_nf_v = val_x_new[n_lf]

    max_l = max([data_pi.shape[1],data_pf.shape[1],data_ni.shape[1],data_nf.shape[1]])

    if data_pi.shape[1] < max_l:
        data_pi = np.concatenate((data_pi.values,
                                 np.zeros((len(data_pi), int(max_l - data_pi.shape[1])))),
                                  axis = 1)
        data_pi_t = np.concatenate((
            data_pi_t.values, np.zeros((len(data_pi_t), int(max_l - data_pi_t.shape[1])))), axis = 1)
        data_pi_v = np.concatenate((
            data_pi_v.values, np.zeros((len(data_pi_v), int(max_l - data_pi_v.shape[1])))), axis = 1)
    if data_ni.shape[1] < max_l:
        data_ni = np.concatenate((data_ni.values,
                                 np.zeros((len(data_ni), int(max_l - data_ni.shape[1])))),
                                  axis = 1)
        data_ni_t = np.concatenate((
            data_ni_t.values, np.zeros((len(data_ni_t), int(max_l - data_ni_t.shape[1])))), axis = 1)
        data_ni_v = np.concatenate((
            data_ni_v.values, np.zeros((len(data_ni_v), int(max_l - data_ni_v.shape[1])))), axis = 1)
    if data_nf.shape[1] < max_l:
        data_nf = np.concatenate((data_nf.values,
                                 np.zeros((len(data_nf), int(max_l - data_nf.shape[1])))),
                                  axis = 1)
        data_nf_t = np.concatenate((
            data_nf_t.values, np.zeros((len(data_nf_t), int(max_l - data_nf_t.shape[1])))), axis = 1)

        data_nf_v = np.concatenate((
            data_nf_v.values, np.zeros((len(data_nf_v), int(max_l - data_nf_v.shape[1])))), axis = 1)

    if data_pf.shape[1] < max_l:
        data_pf = np.concatenate((data_pf.values,
                                 np.zeros((len(data_pf), int(max_l - data_pf.shape[1])))),
                                  axis = 1)
        data_pf_t = np.concatenate((
            data_pf_t.values, np.zeros((len(data_pf_t), int(max_l - data_pf_t.shape[1])))), axis = 1)

        data_pf_v = np.concatenate((
            data_pf_v.values, np.zeros((len(data_pf_v), int(max_l - data_pf_v.shape[1])))), axis = 1)
    ros = SMOTE(random_state=0, sampling_strategy='auto')

    # train_xn1, train_yn1 = ros.fit_resample(data_pi, train_y)
    # train_xn2, train_yn2 = ros.fit_resample(data_ni, train_y)
    # train_xn3, train_yn3 = ros.fit_resample(data_pf, train_y)
    # train_xn4, train_yn4 = ros.fit_resample(data_nf, train_y)
    #
    # train_x = np.stack((train_xn1, train_xn2, train_xn3, train_xn4), axis=2)
    # train_y = train_yn1

    train_x = np.stack((data_pi, data_ni, data_pf, data_nf), axis=2)
    test_x = np.stack((data_pi_t, data_ni_t, data_pf_t, data_nf_t), axis=2)
    val_x = np.stack((data_pi_v, data_ni_v, data_pf_v, data_nf_v), axis=2)


    return train_x,train_y,test_x,test_y,val_x,val_y, p_li,n_li,p_lf,n_lf

def multi_view_Kmeans(train_x, test_x, train_y, test_y, val_x, val_y, header_name, class_n):

    model = ExtraTreesClassifier()
    model.fit(train_x, train_y)
    d = model.feature_importances_
    tmp = zip(range(len(d)), d)
    large5 = heapq.nlargest(60, tmp, key = lambda x:x[1])
    index_new = []
    for i in range(len(large5)):
        index_new.append(large5[i][0])
    train_x = train_x[:, index_new]
    test_x = test_x[:, index_new]
    val_x = val_x[:, index_new]
    # clf = DBSCAN(eps=0.5, min_samples=3)
    clf = KMeans(n_clusters=4)
    clf.fit(np.transpose(train_x))  # 分组
    # print(clf.labels_)
    max_l = max(len(np.where(clf.labels_ == 0)[0]),len(np.where(clf.labels_ == 1)[0]),
                len(np.where(clf.labels_ == 2)[0]),len(np.where(clf.labels_ == 3)[0]))

    if len(np.where(clf.labels_ == 0)[0]) < max_l:
        train_x0 = np.concatenate((train_x[:,np.where(clf.labels_ == 0)],
                                   np.zeros((len(train_x), 1, max_l-len(np.where(clf.labels_ == 0)[0])))),
                                   axis=2)
        test_x0 = np.concatenate((test_x[:,np.where(clf.labels_ == 0)],
                                   np.zeros((len(test_x), 1, max_l-len(np.where(clf.labels_ == 0)[0])))),
                                   axis=2)
        val_x0 = np.concatenate((val_x[:,np.where(clf.labels_ == 0)],
                                   np.zeros((len(val_x), 1, max_l-len(np.where(clf.labels_ == 0)[0])))),
                                   axis=2)
    else:
        train_x0 = train_x[:,np.where(clf.labels_ == 0)]
        test_x0 = test_x[:, np.where(clf.labels_ == 0)]
        val_x0 = val_x[:, np.where(clf.labels_ == 0)]

    if len(np.where(clf.labels_ == 1)[0]) < max_l:
        train_x1 = np.concatenate((train_x[:,np.where(clf.labels_ == 1)],
                                   np.zeros((len(train_x), 1, max_l-len(np.where(clf.labels_ == 1)[0])))),
                                   axis=2)
        test_x1 = np.concatenate((test_x[:,np.where(clf.labels_ == 1)],
                                   np.zeros((len(test_x), 1, max_l-len(np.where(clf.labels_ == 1)[0])))),
                                   axis=2)
        val_x1 = np.concatenate((val_x[:,np.where(clf.labels_ == 1)],
                                   np.zeros((len(val_x), 1, max_l-len(np.where(clf.labels_ == 1)[0])))),
                                   axis=2)
    else:
        train_x1 = train_x[:,np.where(clf.labels_ == 1)]
        test_x1 = test_x[:, np.where(clf.labels_ == 1)]
        val_x1 = val_x[:, np.where(clf.labels_ == 1)]

    if len(np.where(clf.labels_ == 2)[0]) < max_l:
        train_x2 = np.concatenate((train_x[:,np.where(clf.labels_ == 2)],
                                   np.zeros((len(train_x), 1, max_l-len(np.where(clf.labels_ == 2)[0])))),
                                   axis=2)
        test_x2 = np.concatenate((test_x[:,np.where(clf.labels_ == 2)],
                                   np.zeros((len(test_x), 1, max_l-len(np.where(clf.labels_ == 2)[0])))),
                                   axis=2)
        val_x2 = np.concatenate((val_x[:,np.where(clf.labels_ == 2)],
                                   np.zeros((len(val_x), 1,max_l-len(np.where(clf.labels_ == 2)[0])))),
                                   axis=2)
    else:
        train_x2 = train_x[:,np.where(clf.labels_ == 2)]
        test_x2 = test_x[:, np.where(clf.labels_ == 2)]
        val_x2 = val_x[:, np.where(clf.labels_ == 2)]

    if len(np.where(clf.labels_ == 3)[0]) < max_l:
        train_x3 = np.concatenate((train_x[:,np.where(clf.labels_ == 3)],
                                   np.zeros((len(train_x), 1, max_l-len(np.where(clf.labels_ == 3)[0])))),
                                   axis=2)
        test_x3 = np.concatenate((test_x[:,np.where(clf.labels_ == 3)],
                                   np.zeros((len(test_x), 1, max_l-len(np.where(clf.labels_ == 3)[0])))),
                                   axis=2)
        val_x3 = np.concatenate((val_x[:,np.where(clf.labels_ == 3)],
                                   np.zeros((len(val_x), 1, max_l-len(np.where(clf.labels_ == 3)[0])))),
                                   axis=2)
    else:
        train_x1 = train_x[:,np.where(clf.labels_ == 3)]
        test_x1 = test_x[:, np.where(clf.labels_ == 3)]
        val_x1 = val_x[:, np.where(clf.labels_ == 3)]

    train_x = np.concatenate((train_x0, train_x1,train_x2,train_x3), axis=1)
    test_x = np.concatenate((test_x0, test_x1, test_x2, test_x3), axis=1)
    val_x = np.concatenate((val_x0, val_x1, val_x2, val_x3), axis=1)
    train_x = np.transpose(train_x,(0,2,1))
    test_x = np.transpose(test_x, (0, 2, 1))
    val_x = np.transpose(val_x, (0, 2, 1))

    return train_x,train_y,test_x,test_y,val_x,val_y, \
           np.where(clf.labels_ == 0)[0], \
           np.where(clf.labels_ == 1)[0],\
           np.where(clf.labels_ == 2)[0],\
           np.where(clf.labels_ == 3)[0]