from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import scale,Imputer
from hyperopt import fmin,tpe,hp,rand



def oneClassSVM_hp_all(data_scale,labels):
    trainX, testX, trainY, testY = train_test_split(data_scale, labels, random_state=42)
    def oneClassSVM_hp(params):
        kernel = params["kernel"]
        nu = params["nu"]
        gamma = params["gamma"]

        global count
        count = count + 1
        ocSVM_model = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        ocSVM_model.fit(trainX)
        prediction = ocSVM_model.predict(testX).tolist()
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
            "第{}次，score值为{},此时的kernel = {},nu = {},gamma = {}".format(count, f1_score, kernel,
                                                                             nu, gamma))
        return -f1_score

    parameter_space_ocSVM = {
        # loguniform表示该参数取对数后符合均匀分布
        'kernel': hp.choice("kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
        'nu': hp.uniform('nu', 0.00001,0.9999),
        'gamma': hp.uniform("gamma", 0.00001,0.9999),
    }

    best = fmin(oneClassSVM_hp, space=parameter_space_ocSVM, algo=tpe.suggest, max_evals=80)
    best["kernel"] = ['linear', 'poly', 'rbf', 'sigmoid'][best["kernel"]]
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
    # oneClassSVM_hp_all(data_scale,labels)

    trainX, testX, trainY, testY = train_test_split(data_scale, labels, random_state=42)

    kernel = 'sigmoid'
    nu = 0.27764471596021023
    gamma = 0.4193573204183733

    ocSVM_model = svm.OneClassSVM(kernel=kernel,nu=nu,gamma=gamma)
    ocSVM_model.fit(trainX)
    prediction = ocSVM_model.predict(testX).tolist()
    prediction_final = []
    for i in range(len(prediction)):
        if prediction[i] == 1:
            prediction_final.append(0)
        else:
            prediction_final.append(1)

    print(classification_report(testY,prediction_final))
    print(confusion_matrix(testY,prediction_final))