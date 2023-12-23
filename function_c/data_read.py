import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def tail_del(df, rate):
    dele_list = []
    len_1 = df.shape[0]
    for col in df.columns:
        if len(df[col].unique()) < 5:
            if df[col].value_counts().max() / len_1 >= rate:
                dele_list.append(col)
                print("{} 变量分布不均衡，该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df, dele_list

def read_nametxt(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        tmp_list = row.split(': ')  # 按‘，’切分每行的数据
        tmp_list[-1] = tmp_list[-1].replace('.\n', '')  # 去掉换行符
        data.append(tmp_list)  # 将每行数据插入data中
    data = data[1:len(data) + 1]
    data.append(['target', 'symbolic'])
    return data


def data_read(data_name):
    if data_name == 'adult':
        data_ = pd.read_csv('data/adult.csv')
        for i in data_.columns:
            data_[i] = LabelEncoder().fit_transform(data_[i].values)
        epoch, batch_size = 60, 300
        target_name = 'income'

    if data_name == 'bank':
        data_ = pd.read_csv('data/train_bank.csv')
        epoch, batch_size = 60, 300
        data_.drop(['Unnamed: 0', 'loan_id', 'user_id'], axis=1, inplace=True)
        target_name = 'isDefault'

    if data_name == 'german':
        data_ = pd.read_csv('./data/german.csv')
        epoch, batch_size = 300, 50
        target_name = 'Y'
        for i in data_.columns:
            data_[i] = LabelEncoder().fit_transform(data_[i].values)

    if data_name == 'taiwan':
        data_ = pd.read_excel('./data/taiwan.xls', sheet_name='Data', header=0)
        data_.drop(0, axis=0, inplace=True)
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_ = data_.infer_objects()
        target_name = 'Y'
        epoch, batch_size = 60, 300

    if data_name == 'covertype':
        data_ = pd.read_csv('data/' + data_name + '.csv')
        target_name = 'Cover_Type'
        epoch, batch_size = 60, 300
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_, _ = tail_del(data_, rate=0.98)

    if data_name == 'bank_marketing':
        data_name = 'bank-additional-full.csv'  # bank-additional-full.csv
        data_ = pd.read_csv('data/' + data_name, sep=';')
        for col in data_.columns:
            if data_[col].dtype == 'object':
                if data_[col].nunique() >= 0.90 * data_.shape[0]:
                    data_.drop([col], axis=1, inplace=True)
                    print(col, 'is deleted')
                else:
                    data_[col] = LabelEncoder().fit_transform(data_[col])
        target_name = 'y'
        epoch, batch_size = 60, 300

    if data_name == 'CVD':
        data_ = pd.read_csv('data/CVD_cleaned.csv')
        for i in data_.columns:
            if data_[i].dtype == 'object':
                data_[i] = LabelEncoder().fit_transform(data_[i].values)
        epoch, batch_size = 100, 60  # 根据性能差异可调整为 60， 300
        target_name = 'Diabetes'

    if data_name == 'taiwan_credit':  # bank-additional-full.csv CVD_cleaned.csv
        data_ = pd.read_csv('data/' + data_name + '.csv')
        target_name = 'default '
        data_ = data_.drop(['ID'], axis=1, inplace=False)
        for col in data_.columns:
            if data_[col].dtype == 'object':
                data_[col] = LabelEncoder().fit_transform(data_[col])
        epoch, batch_size = 60, 300

    if data_name == 'Intrusion':
        data_name = np.array(read_nametxt('./data/' + data_name + '/kddcup.names'))
        df = pd.read_csv('data/Intrusion/kddcup.data_10_percent.gz', names=data_name[:, 0])
        for k in df.columns:
            if df[k].dtypes == 'object':
                df[k] = LabelEncoder().fit_transform(df[k])
        target_name = 'target'
        epoch, batch_size = 60, 300
        data_, _ = tail_del(df, rate=0.98)

    if 'Unnamed: 0' in data_.columns:
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)

    data = data_.drop([target_name], axis=1)
    x = (data-data.min())/(data.max()-data.min())
    y = data_[target_name]
    return x, y, target_name, epoch, batch_size
