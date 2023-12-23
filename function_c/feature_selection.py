from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

def feature_selection(train_x, train_y, threshold=0.01):
    """
    特征筛选
    :param train_x: 训练集特征
    :param train_y: 训练集标签
    :return: 选取的特征
    """
    clf = ExtraTreesClassifier()
    clf.fit(train_x, train_y)
    feature_importances = pd.Series(clf.feature_importances_, index=train_x.columns)
    feature_importances.sort_values(ascending=False, inplace=True)

    # 选取重要性大于阈值的特征
    selected_features = feature_importances[feature_importances > threshold].index.tolist()
    return selected_features

