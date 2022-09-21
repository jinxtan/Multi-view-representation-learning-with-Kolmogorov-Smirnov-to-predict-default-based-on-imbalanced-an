# -*- coding: utf-8 -*-
"""
Created on 2021/5/12 10:15
@author: Jinxtan
@Institution: Fudan University, Shanghai, China
@email: tanyd20@fudan.edu.cn
@File: ks_curve.py
"""
from pandas import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

linewidth_grid = 0.5
linewidth_axis = 2.0
len_figure = 8
width_figure = 5
def ROC_curve(y_test,y_pred,i):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # 计算fpr,tpr,thresholds
    plt.rcParams['figure.figsize'] = (len_figure, width_figure)
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 15
    plt.grid(ls='--')
    plt.plot(fpr, tpr, linewidth=linewidth_axis)
    plt.title('ROC curve')
    plt.show()

def KS_curve(preds, labels, n, asc):

    # preds is score: asc=1
    # preds is prob: asc=0
    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))
    plt.rcParams['figure.figsize'] = (len_figure, width_figure)
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 15
    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
             color='blue', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
             color='red', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.ks, label='ks',
             color='green', linestyle='-', linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
    plt.legend()
    # plt.show()
    return ksds


def loss_curve(Training,multi_view_way):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['figure.figsize'] = (len_figure, width_figure)
    plt.rcParams['font.size'] = 15
    plt.grid(ls='--')
    plt.plot(Training.history['loss'], label='Training loss')
    plt.plot(Training.history['val_loss'], label='Test loss')
    plt.title('The loss of ' + multi_view_way)
    plt.legend(loc="lower right")
    plt.show()

def acc_curve(Training, multi_view_way):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['figure.figsize'] = (len_figure, width_figure)
    plt.rcParams['font.size'] = 15
    plt.grid(ls='--')
    plt.plot(Training.history['accuracy'], label='Training accuracy',linewidth=linewidth_axis)
    plt.plot(Training.history['val_accuracy'], label='Test accuracy',linewidth=linewidth_axis)
    plt.title('The loss of ' + multi_view_way)
    plt.legend(loc="lower right")
    plt.show()


def KS_train_curve(Training, multi_view_way):
    plt.rc('font', family='Times New Roman')
    plt.rcParams['figure.figsize'] = (len_figure, width_figure)
    plt.rcParams['font.size'] = 15
    plt.grid(ls='--')
    plt.plot(Training.history['ks'], label='Training ks',linewidth=linewidth_axis)
    plt.plot(Training.history['val_ks'], label='Test ks',linewidth=linewidth_axis)

    plt.legend(loc="lower right")
    plt.title('The loss of ' + multi_view_way)
    plt.show()
