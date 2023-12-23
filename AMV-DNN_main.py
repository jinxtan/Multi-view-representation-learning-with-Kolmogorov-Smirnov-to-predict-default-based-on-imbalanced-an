import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from models.multi_view_representation import multi_view_representation
from function_c.data_read import data_read
from function_c.feature_selection import feature_selection
from models.KS_metrics import ks
from models.MV_DNN import MV_DNN
import tensorflow as tf
import numpy as np

import warnings

warnings.filterwarnings('ignore')

data_name = 'bank'

x, y, label_col, batch_size, epoch = data_read(data_name)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
selected_features = feature_selection(train_x, train_y, threshold=0.02)
train_x = train_x[selected_features]
test_x = test_x[selected_features]

# 基于method设置不同的分组方式
group1,group2,group3,group4 = multi_view_representation(train_x,train_y,label_col,method='random')
train_x1,train_x2,train_x3,train_x4 = train_x[group1],train_x[group2],train_x[group3],train_x[group4]
test_x1,test_x2,test_x3,test_x4 = test_x[group1],test_x[group2],test_x[group3],test_x[group4]

model = MV_DNN(group1,group2,group3,group4)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', ks])

call_back = [tf.keras.callbacks.EarlyStopping(monitor='val_ks', patience=20, mode='max'),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_ks', patience=10, mode='max')]

model.fit([train_x1,train_x2,train_x3,train_x4], train_y, batch_size=batch_size, epochs=epoch, callbacks=call_back,
            validation_data=([test_x1,test_x2,test_x3,test_x4], test_y))

pred_porb = model.predict([test_x1,test_x2,test_x3,test_x4])
pred = np.where(pred_porb > 0.5, 1, 0)

print(classification_report(test_y, pred))
print('AUC:', roc_auc_score(test_y, pred_porb))
fpr, tpr, thresholds = roc_curve(test_y, pred_porb)
print('KS:', np.max(tpr - fpr))
