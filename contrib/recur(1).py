import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


DEBUG = True
DEBUG_WIND_NUMBER = 8


###################
# Read raw dataset
###################
raw_df = pd.read_csv("./data/dataset.csv") # Load Dataset
param_df = pd.read_csv("./data/parameters.csv", index_col="风机编号").T # Load turbine parameters
raw_df["label"] = 0
raw_df_index = raw_df.index


##############
# Parameters
##############
# # 完全复现
# ws_interval_width = 0.5 #(m/s)
# pw_interval_width_ratio = 0.0125 # ratio to rated power
# horizontal_low_tolarance = 1.5
# horizontal_high_tolarance = 1.5
# vertical_tolarance = 1.5
# MinPoints = 5
# epsilon_ratio = 0.025

# 对每个风机设置不同的参数
recur_param_df = pd.read_csv("./data/recur_param.csv", index_col="风机编号")

''
#################################################################################
# Augment dataset for double manifold
# 问题描述：有些风机的p-v曲线顶上的那一段水平直线（达到额定功率）数据点太少，会出问题
# 解决思路：过采样，无中生有一些数据点，增强顶上水平直线中的数据点
#################################################################################
def pointGeneration(src_df):
    # 功率上随机扰动
    sigma = src_df['Power'].std()
    src_df['Power'] = src_df['Power'].apply(lambda x: x + random.gauss(0, sigma*0.2))
    # 风速上“延长长度”
    src_df['WindSpeed'] = src_df['WindSpeed'].apply(lambda x: x + random.random()*8)
    # 转速上随机扰动
    sigma = src_df['RotorSpeed'].std()
    src_df['RotorSpeed'] = src_df['RotorSpeed'].apply(lambda x: x + random.gauss(0, sigma*0.2))
    return src_df

print("Augment dataset for p-v curve fitting...")
df = raw_df[raw_df["label"]==0]
ready_winds = [11]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    rated_power = param_df.loc["额定功率", wind_number]
    top_power_rate = recur_param_df.loc[wind_number, "top_power_rate"]
    samples = sub_df[sub_df['Power'] > rated_power * top_power_rate]
    if(wind_number not in ready_winds):
        print("  WindNumber:", wind_number)
        TIMES = ((1-top_power_rate)*len(sub_df) + 0.5) // len(sub_df[sub_df['Power'] > rated_power * top_power_rate]) + 1
        for i in range(round(TIMES.astype(int))):
            # 随机生成
            raw_df = raw_df.append(pointGeneration(samples.copy()), ignore_index=True)
            # # 直接复制
            # raw_df = raw_df.append(samples, ignore_index=True)
print("  raw_df size::")
print(raw_df.loc[raw_df_index, "label"].value_counts())
print("  oversampled_df size:")
print(raw_df["label"].value_counts())


##################################################################################
# 0. Preliminary Elimination with simple rules
#   Remove points whose windspeed, rotorspeed or power < 0.
#   Remove points whose power > 0 but windspeed not in [切入风速, 切出风速] range.
##################################################################################
print("Preliminary Elimination...")
df = raw_df
raw_df.loc[df["Power"]<0, "label"] = 1
raw_df.loc[df["WindSpeed"]<0, "label"] = 1
raw_df.loc[df["RotorSpeed"]<0, "label"] = 1
# 下面这些好像不太行，但是符合论文中的物理规则？
# for wind_number, sub_df in df.groupby("WindNumber"):
#     print("  Wind Number:", wind_number)
#     cut_in_windspeed, cut_out_windspeed = param_df.loc["切入风速", wind_number], param_df.loc["切出风速", wind_number]
#     power_abnormal_condition = (df["Power"] > 0) & ((df["WindSpeed"] < cut_in_windspeed) | (df["WindSpeed"] > cut_out_windspeed))
#     raw_df.loc[power_abnormal_condition, "label"] = 1
print(raw_df.loc[raw_df_index, "label"].value_counts())


#######################################################################################
# 1. The Elimination of Horizontal Sparse Outliers Using Quartile Method
#   For each wind turbind:
#     Divide wind power values into some equal intervals.
#     The quartile method is applied to the wind speed dataset in each power interval.
#     The wind speed data beyond [Fl, Fu] are eliminated from the dataset.
#       IQR = P3−P1 (0.75 and 0.25 percentile points)
#       [Fl,Fu] = [P1 −1.5*IQR, P3 +1.5*IQR]
#######################################################################################
print("Horizontal Eliminating...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    rated_power = param_df.loc["额定功率", wind_number]
    try:
        pw_interval_width_ratio = recur_param_df.loc[wind_number, "pw_interval_width_ratio"]
        horizontal_low_tolarance = recur_param_df.loc[wind_number, "horizontal_low_tolarance"]
        horizontal_high_tolarance = recur_param_df.loc[wind_number, "horizontal_high_tolarance"]
    except:
        pw_interval_width_ratio = 0.0125
        horizontal_low_tolarance = 1.5
        horizontal_high_tolarance = 1.5
    pw_interval_width = pw_interval_width_ratio * rated_power
    sub_df.loc[:, "pw_interval"] = sub_df["Power"].apply(lambda x: x // pw_interval_width)
    for pw_interval, interval_df in sub_df.groupby("pw_interval"):
        p1, p3 = interval_df["WindSpeed"].quantile(0.25), interval_df["WindSpeed"].quantile(0.75)
        iqr = p3 - p1
        fl, fu = p1 - horizontal_low_tolarance*iqr, p3 + horizontal_high_tolarance*iqr
        bad_interval_index = (interval_df["WindSpeed"] < fl) | (interval_df["WindSpeed"] > fu)
        sparse_outlier_index = interval_df[bad_interval_index].index
        raw_df.loc[sparse_outlier_index, "label"] = 1
print(raw_df.loc[raw_df_index, "label"].value_counts())


#############################################################################################################################
# 2. The Elimination of Vertical Sparse Outliers Using Quartile Method
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The quartile method is applied to the wind power dataset in each wind speed interval.
#   Attention: Only the wind power data above Fu are eliminated from the dataset while the data below Fl are not considered.
# #############################################################################################################################
print("Vertical Eliminating...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    try:
        ws_interval_width = recur_param_df.loc[wind_number, "ws_interval_width"]
        vertical_tolarance = recur_param_df.loc[wind_number, "vertical_tolarance"]
    except:
        ws_interval_width = 0.5 # m/s
        vertical_tolarance = 1.5
    sub_df.loc[:, "ws_interval"] = sub_df["WindSpeed"].apply(lambda x: x // ws_interval_width)
    for ws_interval, interval_df in sub_df.groupby("ws_interval"):
        p1, p3 = interval_df["Power"].quantile(0.25), interval_df["Power"].quantile(0.75)
        iqr = p3 - p1
        fl, fu = p1 - 1.5*iqr, p3 + vertical_tolarance*iqr
        bad_interval_index = interval_df["Power"] > fu
        sparse_outlier_index = interval_df[bad_interval_index].index
        raw_df.loc[sparse_outlier_index, "label"] = 1
print(raw_df.loc[raw_df_index, "label"].value_counts())


####################################################################################################################
# 3. The Elimination of Stacked Outliers Using DBSCAN
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The DBSCAN clustering method is applied to the wind power dataset in each wind speed interval.
#     The topmost cluster with largest average power value is the normal data, while other clusters are eliminated.
####################################################################################################################
print("DBSCAN...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    rated_power = param_df.loc["额定功率", wind_number]
    try:
        ws_interval_width = recur_param_df.loc[wind_number, "ws_interval_width"]
        epsilon_ratio = recur_param_df.loc[wind_number, "epsilon_ratio"]
        MinPoints = recur_param_df.loc[wind_number, "MinPoints"]
    except:
        ws_interval_width = 0.5 # m/s
        epsilon_ratio = 0.025
        MinPoints = 5
    epsilon = epsilon_ratio * rated_power
    sub_df.loc[:, "ws_interval"] = sub_df["WindSpeed"].apply(lambda x: x // ws_interval_width)
    for ws_interval, interval_df in sub_df.groupby("ws_interval"):
        X = interval_df["Power"].values.reshape(-1, 1)
        y_pred = DBSCAN(eps=epsilon, min_samples=MinPoints).fit_predict(X)
        cluster_labels = np.unique(y_pred)
        cluster_power = {}
        for cluster_label in cluster_labels:
            if cluster_label == -1:  # out of all clusters
                continue
            cluster_index = np.argwhere(y_pred==cluster_label).reshape(-1)
            cluster_power[cluster_label] = interval_df.iloc[cluster_index]["Power"].mean()
        good_cluster, max_power = None, 0
        for cluster_label, mean_power in cluster_power.items():
            if mean_power > max_power:
                max_power = mean_power
                good_cluster = cluster_label
        bad_cluster_index = np.argwhere(y_pred!=good_cluster).reshape(-1)
        stacked_outlier_index = interval_df.iloc[bad_cluster_index].index
        raw_df.loc[stacked_outlier_index, "label"] = 1
        # print("    Interval:", ws_interval)
        # print("      All clusters:", cluster_labels)
        # print("      All clusters power mean values:", cluster_power.values())
        # print("      Good cluster:", good_cluster)
        # print("      Good cluster power mean value:", max_power)
        # print("      Bad cluster index", bad_cluster_index)
        # print("      Stacked outlier index", stacked_outlier_index)
print(raw_df.loc[raw_df_index, "label"].value_counts())


####################################################################################################################
# 4. 针对3号风机和6号风机的修改
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The DBSCAN clustering method is applied to the wind power dataset in each wind speed interval.
#     The topmost cluster with largest average power value is the normal data, while other clusters are eliminated.
####################################################################################################################
print("Specialize for each wind turbine...")
raw_df["diff"] = 0
df = raw_df.loc[raw_df_index]
df = df[df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    '''
    if not (wind_number == 3 or wind_number == 6):
        continue
    x, y = sub_df["WindSpeed"], sub_df["Power"]
    y_fit = np.polyfit(x, y, 2)  # 二次多项式拟合
    y_show = np.poly1d(y_fit)
    print(y_show)
    sub_df['diff'] = sub_df['Power'] - (y_show.coef[0] * ((sub_df["WindSpeed"] - 0.5) ** 2) + y_show.coef[1] * (sub_df["WindSpeed"] - 0.5) + y_show.coef[2])
    outlier_manifold_index = sub_df[(sub_df["diff"] < 0) & (sub_df["WindSpeed"] > 0) & (sub_df["WindSpeed"] <= 10)].index
    raw_df.loc[outlier_manifold_index, "label"] = 1
    '''

    ### wind_number == 7
    if wind_number == 7:   # 去掉过采样
        x, y = sub_df["WindSpeed"], sub_df["Power"]
        y_fit = np.polyfit(x, y, 2)
        y_show = np.poly1d(y_fit)
        print(y_show)
        sub_df['diff'] = sub_df['Power'] - (y_show.coef[0] * ((sub_df["WindSpeed"] - 1.5) ** 2) + y_show.coef[1] * (sub_df["WindSpeed"] - 1.5) + y_show.coef[2])
        outlier_manifold_index = sub_df[(sub_df["diff"] < 0) & (sub_df["WindSpeed"] > 5) & (sub_df["WindSpeed"] <= 10)].index
        raw_df.loc[outlier_manifold_index, "label"] = 1
    ### wind_number == 8
    if wind_number == 8:
        x, y = sub_df["WindSpeed"], sub_df["Power"]
        y_fit = np.polyfit(x, y, 3)
        y_show = np.poly1d(y_fit)
        print(y_show)
        sub_df['diff'] = sub_df['Power'] - (y_show.coef[0] * ((sub_df["WindSpeed"] - 0.5) ** 3) + y_show.coef[1] * ((sub_df["WindSpeed"] - 0.5) ** 2) + y_show.coef[2] * (sub_df["WindSpeed"] - 0.5) + y_show.coef[3])
        outlier_manifold_index = sub_df[(sub_df["diff"] < 0) & (sub_df["WindSpeed"] > 11) & (sub_df["WindSpeed"] <= 13)].index
        raw_df.loc[outlier_manifold_index, "label"] = 1
    ### wind_number == 9
    if wind_number == 9:
        x, y = sub_df["WindSpeed"], sub_df["Power"]
        y_fit = np.polyfit(x, y, 3)
        y_show = np.poly1d(y_fit)
        print(y_show)
        sub_df['diff'] = sub_df['Power'] - (y_show.coef[0] * ((sub_df["WindSpeed"] - 1) ** 3) + y_show.coef[1] * ((sub_df["WindSpeed"] - 1) ** 2) + y_show.coef[2] * (sub_df["WindSpeed"] - 1) + y_show.coef[3])
        outlier_manifold_index = sub_df[(sub_df["diff"] < 0) & (sub_df["WindSpeed"] > 6) & (sub_df["WindSpeed"] <= 12)].index
        raw_df.loc[outlier_manifold_index, "label"] = 1

print(raw_df.loc[raw_df_index, "label"].value_counts())


################################################
# Plot the results and save the submission file
################################################
print("Plotting 3D scatter...")
raw_df = raw_df.loc[raw_df_index]

# 画三维散点图
for wind_number, sub_df in raw_df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    fig = plt.figure()
    fig.set_size_inches(30, 30, 30)
    ax = Axes3D(fig)
    ax.set_title("Color stands for label")
    ax0 = ax.scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], sub_df["Power"], c=sub_df["label"])
    ax.set_xlabel("WindSpeed")
    ax.set_ylabel("RotorSpeed")
    ax.set_zlabel("Power")
    fig.colorbar(ax0)
    plt.savefig("./figures/recur/" + str(wind_number) + "_results_scatter.jpg")
    plt.close()
# 画维度两两组合的函数关系
print("Plotting 2D scatter...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(40, 20)
    fig.suptitle("WindNumber: " + str(wind_number))
    axs[0].set_title("W&P")
    axs[0].set_xlabel("WindSpeed")
    axs[0].set_ylabel("Power")
    ax0 = axs[0].scatter(sub_df["WindSpeed"], sub_df["Power"], c=sub_df["RotorSpeed"])
    fig.colorbar(ax0, ax=axs[0])
    axs[1].set_title("W&R")
    axs[1].set_xlabel("WindSpeed")
    axs[1].set_ylabel("RotorSpeed")
    ax1 = axs[1].scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], c=sub_df["Power"])
    fig.colorbar(ax1, ax=axs[1])
    axs[2].set_xlabel("RotorSpeed")
    axs[2].set_ylabel("Power")
    axs[2].set_title("R&P")
    ax2 = axs[2].scatter(sub_df["RotorSpeed"], sub_df["Power"], c=sub_df["WindSpeed"])
    fig.colorbar(ax2, ax=axs[2])
    plt.savefig("./figures/recur/" + str(wind_number) + "_dim_relation.jpg")
    plt.close()

submission_df = raw_df[["WindNumber", "Time", "label"]]
submission_df.to_csv("./results/result.csv", index=False)