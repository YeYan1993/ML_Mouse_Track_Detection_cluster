import numpy as np
import pandas as pd
from Mouse_Track_3000.data_process import Data_process

def feature_engine(seq_len,data_x,data_y,data_t,aim_x,aim_y):
    x = data_x[:seq_len]
    y = data_y[:seq_len]
    t = data_t[:seq_len]

    count = len(x)
    aim_x = float(aim_x)
    aim_y = float(aim_y)

    if len(x) == 1:
        x = np.array([x[0]] * 3)
        y = np.array([y[0]] * 3)
        t = np.array([t[0]] * 3)
    elif len(x) == 2:
        x = np.array([x[0]] + [x[1]] * 2)
        y = np.array([y[0]] + [y[1]] * 2)
        t = np.array([t[0]] + [t[1]] * 2)

    x_min = x.min()
    x_ratio = 1.0 * (x[len(x) - 1] - x[0]) / len(x)
    y_min = y.min()
    y_max = y.max()

    x_diff = x[1:] - x[0:-1]
    y_diff = y[1:] - y[0:-1]
    t_diff = t[1:] - t[0:-1]

    x_diff_std = x_diff.std()
    x_diff_max = x_diff.max()
    x_diff_min = x_diff.min()
    x_diff_skew = ((x_diff ** 3).mean() - 3 * x_diff.mean() * x_diff.var() - x_diff.mean() ** 3) / (
                x_diff.var() ** 1.5 + 0.000000001)

    y_diff_mean = np.fabs(y_diff[y_diff != 0]).mean()
    y_diff_std = y_diff[y_diff != 0].std()

    x_back_num = (x_diff < 0).sum()

    DisPoint = 1.0 * sum((x_diff ** 2 + y_diff ** 2) ** 0.5) / len(x)
    Disx_forlat = sum(x_diff[0:len(x_diff) // 2]) / (sum(x_diff[len(x_diff) // 2:len(x_diff)]) + 0.000000001)

    t_diff_mean = t_diff.mean()
    t_diff_min = t_diff.min()
    t_diff_std = t_diff.std()
    duration_mean = 1.0 * (t[len(t) - 1] - t[0]) / len(x)
    timehalf = np.log1p((t[len(t) // 2] - t[0])) - np.log10(t[len(t) - 1] - t[len(t) // 2])

    xy_diff = (x_diff ** 2 + y_diff ** 2) ** 0.5
    xy_diff_max = xy_diff.max()

    Vxy = np.log1p(xy_diff) - np.log1p(t_diff)
    Vxy_diff = Vxy[1:] - Vxy[0:-1]
    Vxy = Vxy[(Vxy > 0) | (Vxy < 1)]
    Vxy_diff = Vxy_diff[(Vxy_diff > 0) | (Vxy_diff < 1)]
    if len(Vxy) < 1:
        vxy_std = 0
        vxy_mean = 0
        vxyfirst = 0
        vxylast = 0
    else:
        vxy_std = Vxy.std()
        vxy_mean = Vxy.mean()
        vxyfirst = Vxy[0]
        vxylast = Vxy[len(Vxy) - 1]

    if len(Vxy_diff) < 1:
        vxy_diff_median = 0
        vxy_diff_mean = 0
        vxy_diff_std = 0
        vxy_diff_max = 0
    else:
        Vxy_diff.sort()
        vxy_diff_median = (Vxy_diff[len(Vxy_diff) // 2] + Vxy_diff[~len(Vxy_diff) // 2]) * 1.0 / 2
        vxy_diff_mean = Vxy_diff.mean()
        vxy_diff_std = Vxy_diff.std()
        vxy_diff_max = Vxy_diff.max()

    angles = np.log1p(y_diff) - np.log1p(x_diff)
    angle_diff = angles[1:] - angles[0:-1]
    angle_diff = angle_diff[(angle_diff > 0) | (angle_diff < 1)]
    angles = angles[(angles > 0) | (angles < 1)]
    if len(angles) < 1:
        angle_std = 0
        angle_kurt = 0
    else:
        angle_std = angles.std()
        angle_kurt = (angles ** 4).mean() / (angles.var() + 0.000000001)

    if len(angle_diff) == 0:
        angle_diff_mean = 0
        angle_diff_std = 0
    else:
        angle_diff_mean = angle_diff.mean()
        angle_diff_std = angle_diff.std()

    Dis_pt2dst = []
    for i in range(len(x)):
        new_x_single = x[i] - aim_x
        new_y_single = y[i] - aim_y
        dist_pt2dst = (new_x_single ** 2 + new_y_single ** 2) ** 0.5
        Dis_pt2dst.append(dist_pt2dst)
    Dis_pt2dst = np.array(Dis_pt2dst)
    # Dis_pt2dst = ((x - np.array([aim_x] * len(x))) ** 2 +
    #               (y - np.array([aim_y] * len(y))) ** 2) ** 0.5
    Dis_pt2dst_diff = Dis_pt2dst[1:] - Dis_pt2dst[0:-1]
    Dis_pt2dst_diff_max = Dis_pt2dst_diff.max()
    Dis_pt2dst_diff_std = Dis_pt2dst_diff.std()

    # 方向角
    DirectAngle = np.sign(x_diff).astype(int).astype(str).astype(object) + np.sign(y_diff).astype(int).astype(
        str).astype(object)
    ConnectDirectAngle = DirectAngle[1:] + DirectAngle[0:-1]
    angle_upTriangle_num = len(ConnectDirectAngle[ConnectDirectAngle == '111-1'])
    angle_downTriangle_num = len(ConnectDirectAngle[ConnectDirectAngle == '1-111'])

    feat = [count, x_min, x_ratio, y_min, y_max, x_diff_std, x_diff_max, x_diff_min, x_diff_skew, y_diff_mean,
            y_diff_std, x_back_num, DisPoint, Disx_forlat, t_diff_mean, t_diff_min, t_diff_std, duration_mean, timehalf,
            xy_diff_max, vxy_std, vxy_mean, vxyfirst, vxylast, vxy_diff_median, vxy_diff_mean, vxy_diff_max,
            vxy_diff_std, angle_std, angle_kurt, angle_diff_mean, angle_diff_std, Dis_pt2dst_diff_max,
            Dis_pt2dst_diff_std, angle_upTriangle_num, angle_downTriangle_num]

    # feat = list(np.nan_to_num(feat))
    #
    # feat_str_list = [str(item) for item in feat]
    # feat_str = ' '.join(feat_str_list)
    return feat



if __name__ == '__main__':
    input_path = "data/dsjtzs_txfz_training.txt"
    input_names = ["index", "move_data", "target", "label"]
    original_data_process = Data_process(input_path, input_names)
    input_data, label, sequence_lenth = original_data_process.preprocessing_data()
    target_point = original_data_process.target_point
    print("Starting Feature Engineering!")
    feature_map = []
    for i in range(len(input_data)):
        x = input_data[i][:,0]
        y = input_data[i][:,1]
        t = input_data[i][:,2]
        seq_len = sequence_lenth[i]
        aim_x,aim_y = target_point[i].split(",")
        feature_single = feature_engine(seq_len,x,y,t,aim_x,aim_y)
        feature_map.append(feature_single)
    print("Feature End")

    feat_name_all = "count, x_min, x_ratio, y_min, y_max, x_diff_std, x_diff_max, x_diff_min, x_diff_skew, y_diff_mean," \
                "y_diff_std, x_back_num, DisPoint, Disx_forlat, t_diff_mean, t_diff_min, t_diff_std, duration_mean, timehalf," \
                "xy_diff_max, vxy_std, vxy_mean, vxyfirst, vxylast, vxy_diff_median, vxy_diff_mean, vxy_diff_max," \
                "vxy_diff_std, angle_std, angle_kurt, angle_diff_mean, angle_diff_std, Dis_pt2dst_diff_max," \
                "Dis_pt2dst_diff_std, angle_upTriangle_num, angle_downTriangle_num"
    feat_name = [i for i in feat_name_all.split(",")]
    print("Total feature is {}".format(len(feat_name)))
    feature_map = np.array(feature_map)
    feature_map_pd = pd.DataFrame(feature_map,columns=feat_name)
    feature_map_pd.to_csv("Feature_map.csv",index=False)
    label_pd = pd.DataFrame({"label":label})
    label_pd.to_csv("label.csv",index=False)
