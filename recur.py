import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


DEBUG = False
DEBUG_WIND_NUMBER = 1

raw_df = pd.read_csv("./data/dataset.csv") # Load Dataset
param_df = pd.read_csv("./data/parameters.csv", index_col="风机编号").T # Load turbine parameters
raw_df["label"] = 0


##############
# Parameters
##############
ws_interval_width = 0.5 #(m/s)
pw_interval_width_ratio = 0.0125 # ratio to rated power
MinPoints = 5
epsilon_ratio = 0.025


########################################
# 0. Preliminary Elimination
#   Remove points whose wind power < 0.
########################################
print("Preliminary Elimination...")
df = raw_df
raw_df.loc[df["Power"]<0, "label"] = 1
print(raw_df["label"].value_counts())


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
    pw_interval_width = pw_interval_width_ratio * rated_power
    sub_df.loc[:, "pw_interval"] = sub_df["Power"].apply(lambda x: x // pw_interval_width)
    for pw_interval, interval_df in sub_df.groupby("pw_interval"):
        p1, p3 = interval_df["WindSpeed"].quantile(0.25), interval_df["WindSpeed"].quantile(0.75)
        iqr = p3 - p1
        fl, fu = p1 - 1.5*iqr, p3 + 1.5*iqr
        bad_interval_index = (interval_df["WindSpeed"] < fl) | (interval_df["WindSpeed"] > fu)
        sparse_outlier_index = interval_df[bad_interval_index].index
        raw_df.loc[sparse_outlier_index, "label"] = 1
print(raw_df["label"].value_counts())


#############################################################################################################################
# 2. The Elimination of Vertical Sparse Outliers Using Quartile Method
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The quartile method is applied to the wind power dataset in each wind speed interval. 
#   Attention: Only the wind power data above Fu are eliminated from the dataset while the data below Fl are not considered.
#############################################################################################################################
print("Vertical Eliminating...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    sub_df.loc[:, "ws_interval"] = sub_df["WindSpeed"].apply(lambda x: x // ws_interval_width)
    for ws_interval, interval_df in sub_df.groupby("ws_interval"):
        p1, p3 = interval_df["Power"].quantile(0.25), interval_df["Power"].quantile(0.75)
        iqr = p3 - p1
        fl, fu = p1 - 1.5*iqr, p3 + 1.5*iqr
        bad_interval_index = interval_df["Power"] > fu
        sparse_outlier_index = interval_df[bad_interval_index].index
        raw_df.loc[sparse_outlier_index, "label"] = 1
print(raw_df["label"].value_counts())


####################################################################################################################
# 3. The Elimination of Stacked Outliers Using DBSCAN
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The DBSCAN clustering method is applied to the wind power dataset in each wind speed interval.
#     The topmost cluster with largest average power value is the normal data, while other clusters are eliminated.
####################################################################################################################
print("DBSCAN...")
df = raw_df[raw_df["label"]==0]
print(df["label"].value_counts())
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    rated_power = param_df.loc["额定功率", wind_number]
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
print(raw_df["label"].value_counts())


################################################
# Plot the results and save the submission file
################################################
print("Plotting...")
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
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
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
