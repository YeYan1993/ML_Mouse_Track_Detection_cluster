from hyperopt import fmin,tpe,hp,rand
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.preprocessing import scale,Imputer
from sklearn.metrics import silhouette_score  #聚类评估：轮廓系数


def Kmeans_hp_all(data_scale):
    def kmeans_hp(n_clusters):
        global count
        count = count + 1
        labels = KMeans(n_clusters).fit(data_scale).labels_
        si_score = silhouette_score(data_scale, labels)
        print("第{}次，score值为{},此时的n_clusters = {}".format(count, si_score, n_clusters))
        return -si_score

    best = fmin(fn=kmeans_hp, space=hp.choice('n_clusters', [i for i in range(2, 31)]), algo=tpe.suggest, max_evals=50)
    # best["n_clsuters"]返回的是数组下标，因此需要把它还原回来(只有hp.choice是这样的)
    best["n_clusters"] = [i for i in range(2, 11)][best["n_clusters"]]
    print(best)

def DBSCAN_hp_all(data_scale):
    def DBSCAN_hp(params):
        eps = params["eps"]
        min_samples = params["min_samples"]
        metric = params["metric"]
        algorithm = params["algorithm"]
        global count_db
        count_db = count_db + 1
        labels_db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm).fit(data_scale).labels_
        if len(set(labels_db)) == 1:
            print("这里只有一个类！此时的eps = {},min_samples = {},metric = {},algorithm = {}".format(eps, min_samples, metric,
                                                                                           algorithm))
        else:
            si_score_db = silhouette_score(data_scale, labels_db)
            print(
                "第{}次，score值为{},此时的eps = {},min_samples = {},metric = {},algorithm = {}".format(count_db, si_score_db, eps,
                                                                                                min_samples, metric,
                                                                                                algorithm))
            return -si_score_db

    parameter_space_db = {
        # loguniform表示该参数取对数后符合均匀分布
        'eps': hp.uniform("eps", 1, 10),
        'min_samples': hp.choice('min_samples', range(2, 20)),
        'metric': hp.choice("metric", ["euclidean", "manhattan", "chebyshev"]),
        'algorithm': hp.choice("algorithm", ['auto', 'brute']),
    }

    best_db = fmin(DBSCAN_hp, space=parameter_space_db, algo=tpe.suggest, max_evals=100)
    best_db["min_samples"] = [i for i in range(2, 20)][best_db["min_samples"]]
    best_db["metric"] = ["euclidean", "manhattan", "chebyshev"][best_db["metric"]]
    best_db["algorithm"] = ['auto', 'brute'][best_db["algorithm"]]
    print(best_db)


def Birch_hp_all(data_scale):
    def Birch_hp(params):
        threshold = params["threshold"]
        branching_factor = params["branching_factor"]
        n_clusters = params["n_clusters"]

        global count_birch
        count_birch = count_birch + 1
        labels_birch = Birch(threshold=threshold,branching_factor=branching_factor,n_clusters=n_clusters).fit(data_scale).labels_
        if len(set(labels_birch)) == 1:
            print("这里只有一个类！此时的threshold = {},branching_factor = {},n_clusters = {}".format(threshold,branching_factor,n_clusters))
        else:
            si_score_birch = silhouette_score(data_scale, labels_birch)
            print(
                "第{}次，score值为{},此时的threshold = {},branching_factor = {},n_clusters = {}".format(count_birch, si_score_birch, threshold,
                                                                                                branching_factor, n_clusters))
            return -si_score_birch

    parameter_space_birch = {
        # loguniform表示该参数取对数后符合均匀分布
        'threshold': hp.uniform("threshold", 0, 1),
        'branching_factor': hp.choice('branching_factor', range(30, 80)),
        'n_clusters': hp.choice("n_clusters", range(2,10)),
    }

    best_birch = fmin(Birch_hp, space=parameter_space_birch, algo=tpe.suggest, max_evals=100)

    best_birch["branching_factor"] = [i for i in range(30,80)][best_birch["branching_factor"]]
    best_birch["n_clusters"] = [i for i in range(2, 10)][best_birch["n_clusters"]]
    print(best_birch)



if __name__ == '__main__':
    data = pd.read_csv("Feature_map.csv", index_col=False)
    label_origin = pd.read_csv("label.csv", index_col=False)
    np_data = np.array(data)
    result = np.isinf(np_data)
    inf_xy = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j]:
                inf_xy.append((i,j))
    np_data[result] = 0 # 这里由于在其他横排的地方都是0，因此用0代替
    data_scale = scale(np_data.astype(np.float32))
    data_scale = Imputer().fit_transform(data_scale)

    """1.Kmeans_hyperopt"""
    print("Starting Kmeans_hyperopt.......")
    print("###############################")
    count = 0
    Kmeans_hp_all(data_scale)

    """2.DBSCAN_hyperopt"""
    print("Starting DBSCAN_hyperopt.......")
    print("###############################")
    count_db = 0
    DBSCAN_hp_all(data_scale)

    """3.Birch_hyperopt"""
    print("Starting Birch_hyperopt.......")
    print("###############################")
    count_birch = 0
    Birch_hp_all(data_scale)

    #### 结果打印如下
    #### {'n_clusters': 2} ==> Kmeans
    #### {'metric': 'chebyshev', 'algorithm': 'brute', 'eps': 9.811131350399677, 'min_samples': 13} ==> DBSCAN
    #### {'n_clusters': 2, 'threshold': 0.2275991951919598, 'branching_factor': 48} ==> Birch




