import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


def multi_view_representation(train_x, train_y, label_col, method='amv'):
    """
    多视图表示
    :param train_x: 训练集特征
    :param train_y: 训练集标签
    :return: 多视图表示后的特征
    """
    if method == 'amv':
        # 将特征划分为正相关和负相关
        corr = train_x.join(train_y).corr()[label_col].iloc[:-1]
        corr.sort_values(ascending=False, inplace=True)
        positive_features = corr[corr > 0].index.tolist()
        negative_features = corr[corr < 0].index.tolist()

        # 将特征分为离散组和连续组
        nunique = train_x.nunique()
        mean_nunique = nunique.median()
        discrete_features = nunique[nunique <= mean_nunique].index.tolist()
        continuous_features = nunique[nunique > mean_nunique].index.tolist()

        positive_continuous_features = list(set(positive_features).intersection(set(continuous_features)))
        positive_discrete_features = list(set(positive_features).intersection(set(discrete_features)))
        negative_continuous_features = list(set(negative_features).intersection(set(continuous_features)))
        negative_discrete_features = list(set(negative_features).intersection(set(discrete_features)))

        return positive_continuous_features, positive_discrete_features, negative_continuous_features, negative_discrete_features

    elif method == 'kmeans':
        # 将特征划分为正相关和负相关
        clf = KMeans(n_clusters=4)
        clf.fit(np.transpose(train_x))
        labels = clf.labels_
        labels = pd.Series(labels, index=train_x.columns)

        # 将特征根据聚类结果分为4类
        group1 = labels[labels == 0].index.tolist()
        group2 = labels[labels == 1].index.tolist()
        group3 = labels[labels == 2].index.tolist()
        group4 = labels[labels == 3].index.tolist()

        return group1, group2, group3, group4
    elif method == 'random':
        # 将特征列随机划分为四个视野
        features = train_x.columns.tolist()
        np.random.shuffle(features)
        group1 = features[:int(len(features) / 4)]
        group2 = features[int(len(features) / 4):int(len(features) / 2)]
        group3 = features[int(len(features) / 2):int(len(features) * 3 / 4)]
        group4 = features[int(len(features) * 3 / 4):]

        return group1, group2, group3, group4