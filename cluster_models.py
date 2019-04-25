import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.preprocessing import scale,Imputer


def anomaly_Kmeans(data_scale,thread=90):
    kmeans_model = KMeans(n_clusters=2).fit(data_scale)
    centers = kmeans_model.cluster_centers_
    labels_ = kmeans_model.labels_
    label_0 = np.where(labels_ == 0)[0]
    label_1 = np.where(labels_ == 1)[0]
    # print("label 是0的有{}，label 是1的有{}".format(len(label_0),len(label_1)))

    # 求label=0的点和聚类重心的欧式距离
    data_scale_0 = data_scale[label_0]
    data_scale_1 = data_scale[label_1]
    distance_0 = []
    for i in range(len(data_scale_0)):
        dist = np.linalg.norm(data_scale_0[i] - centers[0])
        distance_0.append(dist)
    distance_1 = []
    for i in range(len(data_scale_1)):
        dist = np.linalg.norm(data_scale_1[i] - centers[1])
        distance_1.append(dist)

    anomaly_0 = np.percentile(distance_0, thread)
    anomaly_1 = np.percentile(distance_1, thread)
    anomaly_0_index = np.where(np.array(distance_0) > anomaly_0)[0]
    anomaly_1_index = np.where(np.array(distance_1) > anomaly_1)[0]

    all_anomaly_index = sorted(np.append(label_0[anomaly_0_index],label_1[anomaly_1_index]).tolist())
    return all_anomaly_index

def anomaly_DBSCAN(data_scale):
    dbscan_model = DBSCAN(metric = 'chebyshev', algorithm = 'brute', eps = 9.811131350399677, min_samples = 13).fit(data_scale)
    labels_ = dbscan_model.labels_
    label_0 = np.where(labels_ == 0)[0]
    label_anomaly = np.where(labels_ == -1)[0]
    return label_anomaly

def anomaly_Birch(data_scale):
    birch_model = Birch(n_clusters = 2, threshold = 0.2275991951919598, branching_factor = 48).fit(data_scale)
    labels_ = birch_model.labels_

    label_0 = np.where(labels_ == 0)[0]
    label_1 = np.where(labels_ == 1)[0]
    return label_0



if __name__ == '__main__':
    data = pd.read_csv("Feature_map.csv", index_col=False)
    label_origin = pd.read_csv("label.csv", index_col=False)
    np_data = np.array(data)
    result = np.isinf(np_data)
    inf_xy = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j]:
                inf_xy.append((i, j))
    np_data[result] = 0  # 这里由于在其他横排的地方都是0，因此用0代替
    data_scale = scale(np_data.astype(np.float32))
    data_scale = Imputer().fit_transform(data_scale)


    """1.Kmeans"""
    anomaly_index_km = anomaly_Kmeans(data_scale,thread=93)

    """2.DBSCAN"""
    anomaly_index_db = anomaly_DBSCAN(data_scale)

    """3.Birch"""
    anomaly_index_birch = anomaly_Birch(data_scale)

    anomaly_intersection_1 = list(set(anomaly_index_km).union(set(anomaly_index_birch)))
    anomaly_intersection_2 = list(set(anomaly_intersection_1).union(set(anomaly_index_db)))
    # print(anomaly_index_db)

    true_label = np.array(pd.read_csv("label.csv"))
    true_label_index = np.where(true_label == 1)[0]
    result = list(set(anomaly_intersection_2).intersection(set(true_label_index.tolist())))
    print(len(result))
    print(len(anomaly_intersection_2))
    print(len(true_label_index))
    print(len(result)/len(anomaly_intersection_2))
    print(len(result)/len(true_label_index))