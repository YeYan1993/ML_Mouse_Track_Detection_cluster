from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import scale,Imputer
from hyperopt import fmin,tpe,hp,rand



def isolationForest_hp_all(data_scale,labels):
    trainX, testX, trainY, testY = train_test_split(data_scale, labels, random_state=42)
    def isolationForest_hp(params):
        n_estimators = params["n_estimators"]
        max_samples = params["max_samples"]
        max_features = params["max_features"]

        global count
        count = count + 1
        if_model = IsolationForest(n_estimators=n_estimators,max_samples=500,max_features=max_features,n_jobs=-1,contamination='auto',behaviour='new')
        if_model.fit(trainX)
        prediction = if_model.predict(testX).tolist()
        prediction_final = []
        for i in range(len(prediction)):
            if prediction[i] == 1:
                prediction_final.append(0)
            else:
                prediction_final.append(1)
        c_m = confusion_matrix(testY, prediction_final)
        precision = c_m[1][1] / (c_m[1][1] + c_m[0][1])
        recall = c_m[1][1] / (c_m[1][1] + c_m[1][0])
        f1_score = 2 * precision * recall / (precision + recall)
        print(
            "第{}次，score值为{},此时的n_estimators = {},max_samples = {},max_features = {}".format(count, f1_score, n_estimators,
                                                                                                     max_samples, max_features))
        return -f1_score

    parameter_space_isf = {
        # loguniform表示该参数取对数后符合均匀分布
        'n_estimators': hp.choice("n_estimators", range(400, 900)),
        'max_samples': hp.choice('max_samples', range(256, 800)),
        'max_features': hp.choice("max_features", range(1,10)),
    }

    best = fmin(isolationForest_hp, space=parameter_space_isf, algo=tpe.suggest, max_evals=500)
    best["n_estimators"] = [i for i in range(400, 900)][best["n_estimators"]]
    best["max_samples"] = [i for i in range(256,800)][best["max_samples"]]
    best["max_features"] = [i for i in range(1,10)][best["max_features"]]
    print(best)



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
    labels = np.array(pd.read_csv("label.csv"))
    count = 0
    # isolationForest_hp_all(data_scale,labels)

    trainX, testX, trainY, testY = train_test_split(data_scale, labels, random_state=42)


    if_model = IsolationForest(n_estimators=498,n_jobs=-1,max_samples=467,max_features=1)
    if_model.fit(trainX)
    prediction = if_model.predict(testX).tolist()
    prediction_final = []
    for i in range(len(prediction)):
        if prediction[i] == 1:
            prediction_final.append(0)
        else:
            prediction_final.append(1)

    print(classification_report(testY,prediction_final))
    print(confusion_matrix(testY,prediction_final))






