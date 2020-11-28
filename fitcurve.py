import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D


DEBUG = False
DEBUG_WIND_NUMBER = 3


###################
# Read raw dataset
###################
print("Reading dataset...")
raw_df = pd.read_csv("./data/dataset.csv") # Load Dataset
param_df = pd.read_csv("./data/parameters.csv", index_col="风机编号").T # Load turbine parameters
raw_df["ts"] = raw_df["Time"].apply(lambda x: int(time.mktime(time.strptime(x, "%Y/%m/%d %H:%M"))))
raw_df["label"] = 0
raw_df["selected"] = 0
raw_df_index = raw_df.index


##############
# Parameters
##############
pass


# 对每个风机设置不同的参数
recur_param_df = pd.read_csv("./data/recur_param.csv", index_col="风机编号")


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
# # 下面这些好像不太行，但是符合论文中的物理规则？
# for wind_number, sub_df in df.groupby("WindNumber"):
#     print("  Wind Number:", wind_number)
#     cut_in_windspeed, cut_out_windspeed = param_df.loc["切入风速", wind_number], param_df.loc["切出风速", wind_number]
#     power_abnormal_condition = (df["Power"] > 0) & ((df["WindSpeed"] < cut_in_windspeed) | (df["WindSpeed"] > cut_out_windspeed))
#     raw_df.loc[power_abnormal_condition, "label"] = 1
# 对每个风机的特别预处理
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    pass
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
# 3. The Elimination according to timestamp
#   For each wind turbine:
#     Select points which looks good. 
####################################################################################################################
print("Timestamp check...")
df = raw_df[raw_df["label"]==0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    timestamp = sub_df["Time"].apply(lambda x: int(time.mktime(time.strptime(x, "%Y/%m/%d %H:%M"))))

    if wind_number == 3:
        selected_condition = (sub_df["ts"] > 1.528 * 10**9) & (sub_df["ts"] < 1.535 * 10**9)
        selected_index = sub_df[selected_condition].index
        raw_df.loc[selected_index, "selected"] = 1
        continue

    threshold = sub_df["ts"].quantile(0.90)
    selected_condition = sub_df["ts"] > threshold
    selected_index = sub_df[selected_condition].index
    raw_df.loc[selected_index, "selected"] = 1

print(raw_df.loc[raw_df_index, "label"].value_counts())


###########
# 5. Fit poly curves
################
print("Fitting curves...")
curves = []
df = raw_df[raw_df["selected"]==1]
for wind_number, sub_df in df.groupby("WindNumber"):
       
    print("  Wind Number:", wind_number)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(40, 20)
    fig.suptitle("WindNumber: " + str(wind_number))
    axs[0].set_title("W&P")
    axs[0].set_xlabel("WindSpeed")
    axs[0].set_ylabel("Power")
    axs[0].grid()
    
    if(wind_number == 11 or wind_number == 12):
        z0 = np.polyfit(sub_df["WindSpeed"], sub_df["Power"],9)
    else:
        z0 = np.polyfit(sub_df["WindSpeed"], sub_df["Power"],5)
    p0 = np.poly1d(z0)
    curves.append(p0)
    ax0 = axs[0].scatter(sub_df["WindSpeed"], sub_df["Power"])

    x = np.linspace(2,14, 50)
    y =p0(x)
    axs[0].plot(x,y,color='red')


#####################################
# 6. Eliminate according to curves
#####################################
df = raw_df[raw_df['label'] == 0]
for wind_number, sub_df in df.groupby("WindNumber"):
    
    print("  Wind Number:", wind_number)
    # delete points below the line1: (10.7, 1500) -> (12.3, 2000)
    outlier_condition11 = sub_df["Power"] < curves[wind_number-1](sub_df["WindSpeed"] - delta)
    outlier_condition12 = sub_df["Power"] > curves[wind_number-1](sub_df["WindSpeed"] + delta)
    outlier_condition1 = (outlier_condition11 | outlier_condition12)
    outlier_index = sub_df[outlier_condition1].index    
    raw_df.loc[outlier_index, "label"] = 1


####################################################################################################################
# 7. Specialized processing for each wind turbine...
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
    pass
print(raw_df.loc[raw_df_index, "label"].value_counts())


################################################
# Plot the results and save the submission file
################################################
raw_df = raw_df.loc[raw_df_index]

print("Plotting 3D scatter...")
# 画三维散点图（结果图）
for wind_number, sub_df in raw_df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    fig = plt.figure()
    fig.set_size_inches(30, 30, 30)
    ax = Axes3D(fig)
    ax.set_title("Color stands for label")
    ax0 = ax.scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], sub_df["Power"], c=sub_df["selected"])
    ax.set_xlabel("WindSpeed")
    ax.set_ylabel("RotorSpeed")
    ax.set_zlabel("Power")
    fig.colorbar(ax0)
    plt.savefig("./figures/fitcurve/" + str(wind_number) + "_selected_scatter.jpg")
    plt.close()

# 画维度两两组合的函数关系（结果图）
print("Plotting 2D scatter...")
# df = raw_df[raw_df["label"]==0]
df = raw_df
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
    axs[0].set_xlim(0, 25)
    axs[0].set_xticks(np.linspace(0, 25, 26))
    axs[0].set_ylim(-100, 2300)
    axs[0].set_yticks(np.linspace(0, 2500, 26))
    ax0 = axs[0].scatter(sub_df["WindSpeed"], sub_df["Power"], c=sub_df["selected"])
    fig.colorbar(ax0, ax=axs[0])
    axs[1].set_title("W&R")
    axs[1].set_xlabel("WindSpeed")
    axs[1].set_ylabel("RotorSpeed")
    ax1 = axs[1].scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], c=sub_df["selected"])
    fig.colorbar(ax1, ax=axs[1])
    axs[2].set_xlabel("RotorSpeed")
    axs[2].set_ylabel("Power")
    axs[2].set_title("R&P")
    ax2 = axs[2].scatter(sub_df["RotorSpeed"], sub_df["Power"], c=sub_df["selected"])
    fig.colorbar(ax2, ax=axs[2])
    plt.savefig("./figures/fitcurve/" + str(wind_number) + "_selected_dim_relation.jpg")
    plt.close()

# 画三维散点图
print("Plotting 3D raw scatter...")
for wind_number, sub_df in raw_df.groupby("WindNumber"):
    color = sub_df["ts"]
    if not DEBUG:
        break
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    fig = plt.figure()
    fig.set_size_inches(30, 30, 30)
    ax = Axes3D(fig)
    ax.set_title("Color stands for label")
    ax0 = ax.scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], sub_df["Power"], c=color)
    ax.set_xlabel("WindSpeed")
    ax.set_ylabel("RotorSpeed")
    ax.set_zlabel("Power")
    fig.colorbar(ax0)
    plt.savefig("./figures/fitcurve/" + str(wind_number) + "_raw_scatter.jpg")
    # plt.show()
    plt.close()

# 画维度两两组合的函数关系
print("Plotting 2D raw scatter...")
df = raw_df
for wind_number, sub_df in df.groupby("WindNumber"):
    color = sub_df["ts"]
    if not DEBUG:
        break
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    print("  Wind Number:", wind_number)
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(40, 20)
    fig.suptitle("WindNumber: " + str(wind_number))
    axs[0].set_title("W&P")
    axs[0].set_xlabel("WindSpeed")
    axs[0].set_ylabel("Power")
    axs[0].set_xlim(0, 25)
    axs[0].set_xticks(np.linspace(0, 25, 26))
    axs[0].set_ylim(-100, 2300)
    axs[0].set_yticks(np.linspace(0, 2500, 26))
    ax0 = axs[0].scatter(sub_df["WindSpeed"], sub_df["Power"], c=color)
    fig.colorbar(ax0, ax=axs[0])
    axs[1].set_title("W&R")
    axs[1].set_xlabel("WindSpeed")
    axs[1].set_ylabel("RotorSpeed")
    ax1 = axs[1].scatter(sub_df["WindSpeed"], sub_df["RotorSpeed"], c=color)
    fig.colorbar(ax1, ax=axs[1])
    axs[2].set_xlabel("RotorSpeed")
    axs[2].set_ylabel("Power")
    axs[2].set_title("R&P")
    ax2 = axs[2].scatter(sub_df["RotorSpeed"], sub_df["Power"], c=color)
    fig.colorbar(ax2, ax=axs[2])
    plt.savefig("./figures/fitcurve/" + str(wind_number) + "_raw_dim_relation.jpg")
    plt.close()

if not DEBUG:
    submission_df = raw_df[["WindNumber", "Time", "label", "selected"]]
    submission_df.to_csv("./results/result.csv", index=False)
