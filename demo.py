import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import time, datetime

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.metrics import calinski_harabasz_score
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


df = pd.read_csv("./data/dataset.csv") # 加载风机数据
param_df = pd.read_csv("./data/parameters.csv", index_col="风机编号").T # 加载风机参数
output_df = pd.DataFrame(None, columns=["WindNumber", "Time", "label"])


# 遍历每个风机，每个分别进行预处理、聚类及其他处理
for wind_number, sub_df in df.groupby("WindNumber"):
    # if not wind_number == 8:
    #     continue

    print("\nWindNumber:", wind_number)

    # 加载风机参数（风机编号,额定功率,风轮直径,切入风速,切出风速,风轮转速范围）
    param = param_df[wind_number]

    #################
    # Preprocessing #
    #################
    print("--预处理...")
    # # 去掉WindNumber, Time列
    # sub_df = raw_sub_df.drop(["WindNumber", "Time"], axis=1)
    
    # 清洗：检查WindSpeed和RotorSpeed是否在范围内
    WASH_BY_PARAM = False
    if WASH_BY_PARAM:
        windspeed_lb, windspeed_ub = param['切入风速'], param['切出风速']
        windspeed_col = sub_df["WindSpeed"]
        windspeed_in_range = (windspeed_col < windspeed_ub) & (windspeed_col > windspeed_lb)
        rotorspeed_range = param['风轮转速范围'].split('-')
        rotorspeed_lb, rotorspeed_ub = float(rotorspeed_range[0]), float(rotorspeed_range[1])
        rotorspeed_col = sub_df["RotorSpeed"]
        rotorspeed_in_range = (rotorspeed_col < rotorspeed_ub) & (rotorspeed_col > rotorspeed_lb)
        # print(windspeed_in_range.value_counts())
        # print(rotorspeed_in_range.value_counts())
        inrange_sub_df = sub_df[windspeed_in_range & rotorspeed_in_range] # 按照风机参数去掉范围外的点
    else:
        inrange_sub_df = sub_df

    # 按照时间序列画图
    PLOT_TIME_SERIES = False
    if PLOT_TIME_SERIES:
        print(sub_df["Time"])
        sub_df["Time"] = sub_df["Time"].apply(
            lambda x: time.mktime(time.strptime(x, "%Y/%m/%d %H:%M")) / 100 - 15000000
        )
        fig = plt.figure()
        fig.set_size_inches(1000, 30)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax0 = ax.plot(sub_df["Time"], sub_df["RotorSpeed"])
        ax.set_xlabel("Time")
        ax.set_ylabel("RotorSpeed")
        plt.show()
    
    # 二度清洗，清洗掉Windspeed增加而RotorSpeed和Power都不增加的点
    # 基本思路: 将RotorSpeed和Power分别划分为多个区间，形成网格
    #           对每个网格，记录网格内所有点WindSpeed的范围（极差）
    #           找到WindSpeed行为异常的网格，去掉该网格中WindSpeed的异常点
    # 0920memo: 能不能对每个网格内的子数据集用混合高斯模型，使用模型算出的方差作为异常判断依据，以适应“两条形”的数据集
    # 划分方法分为等深划分和等宽划分
    GRID_VAR = True
    GRID_RANGE = False
    GRID = GRID_VAR or GRID_RANGE
    GRID_METHOD = "equal-width" # equal-width or equal-depth
    if GRID_VAR: # 加载用于grid_var的参数
        grid_param_df = pd.read_csv("./data/ew_grid_var_param.csv", index_col="风机编号")
    elif GRID_RANGE: # 加载用于grid_range的参数
        grid_param_df = pd.read_csv("./data/ew_grid_range_param.csv", index_col="风机编号")
    rs_interval_n, pw_interval_n, rs_lb_min, lb_percentile, ub_percentile, small_grid_coef, k, clip_ratio_step = grid_param_df.loc[wind_number]
    if GRID and GRID_METHOD == "equal-width":
        grid2attr = {}
        # 下面正式开始划分网格
        print("--划分grid_var网格，使用等宽划分...")
        print("--参数")
        print(grid_param_df.loc[wind_number])
        rs, pw = inrange_sub_df["RotorSpeed"], inrange_sub_df["Power"]
        rs_ub, rs_lb, pw_ub, pw_lb = rs.max(), max(rs.min(), -1.), pw.max(), max(pw.min(), rs_lb_min)
        rs_step, pw_step = (rs.max() - rs.min()) / rs_interval_n, (pw.max() - pw.min()) / pw_interval_n
        print("  rs_lb:", rs_lb, "rs_ub:", rs_ub, "rs_step:", rs_step)
        print("  pw_lb:", pw_lb, "pw_ub:", pw_ub, "pw_step:", pw_step)
        interval_df = inrange_sub_df.copy()
        # 划分网格，记录每个(RotorSpeed, Power)网格内的WindSpeed上下限和方差
        interval_df["RotorSpeed"] = interval_df["RotorSpeed"].apply(lambda x: (x - rs_lb) // rs_step)
        interval_df["Power"] = interval_df["Power"].apply(lambda x: (x - pw_lb) // pw_step)
        bound_var_df = pd.DataFrame([], columns=["grid_lb", "grid_ub", "grid_var"])
        for rs_pw, ws_df in interval_df.groupby(["RotorSpeed", "Power"]):
            # 取百分数作为网格内的WindSpeed上下限
            ws_ub = ws_df["WindSpeed"].quantile(ub_percentile)
            if lb_percentile >= 0.:
                ws_lb = ws_df["WindSpeed"].quantile(lb_percentile)
            else: # 百分数为负的情况
                ws_min = ws_df["WindSpeed"].quantile(0.)
                ws_symmetric = ws_df["WindSpeed"].quantile(-lb_percentile)
                ws_lb = ws_min - (ws_symmetric - ws_min)
            ws_var = ws_df["WindSpeed"].var()
            if len(ws_df) < small_grid_coef * len(interval_df) / (rs_interval_n * pw_interval_n):
                ws_var = 0.
            rs_grid_series = pd.Series(rs_pw[0], name="rs_grid", index=ws_df.index)
            pw_grid_series = pd.Series(rs_pw[1], name="pw_grid", index=ws_df.index)
            lb_series = pd.Series(ws_lb, name="grid_lb", index=ws_df.index)
            ub_series = pd.Series(ws_ub, name="grid_ub", index=ws_df.index)
            var_series = pd.Series(ws_var, name="grid_var", index=ws_df.index)
            bound_var_df = bound_var_df.append(
                pd.concat([rs_grid_series, pw_grid_series, lb_series, ub_series, var_series], axis=1)
            )
            grid2attr[rs_pw] = {"lb": ws_lb, "ub": ws_ub, "var": ws_var, "n": len(ws_df)}
        inrange_sub_df = pd.concat([inrange_sub_df, bound_var_df], axis=1)
    if GRID and GRID_METHOD == "equal-depth":
        grid2attr = {}
        # 下面正式开始划分网格
        print("--划分grid_var网格，使用等深划分...")
        print("--参数")
        print(grid_param_df.loc[wind_number])
        rs, pw = inrange_sub_df["RotorSpeed"], inrange_sub_df["Power"]
        interval_df = inrange_sub_df.copy()
        interval_df["rs_grid"] = None
        interval_df["pw_grid"] = None
        # 划分网格，记录每个(RotorSpeed, Power)网格内的WindSpeed上下限和方差
        for i in range(int(rs_interval_n)):
            grid_rs_lb = rs.quantile(i / rs_interval_n)
            grid_rs_ub = rs.quantile((i+1) / rs_interval_n)
            rs_in_grid = (rs <= grid_rs_ub) & (rs >= grid_rs_lb)
            interval_df["rs_grid"][rs_in_grid] = i
        for i in range(int(pw_interval_n)):
            grid_pw_lb = pw.quantile(i / pw_interval_n)
            grid_pw_ub = pw.quantile((i+1) / pw_interval_n)
            pw_in_grid = (pw <= grid_pw_ub) & (pw >= grid_pw_lb)
            interval_df["pw_grid"][pw_in_grid] = i
        bound_var_df = pd.DataFrame([], columns=["grid_lb", "grid_ub", "grid_var"])
        for rs_pw, ws_df in interval_df.groupby(["rs_grid", "pw_grid"]):
            # 取一定的百分数作为网格内的WindSpeed上下限
            ws_ub = ws_df["WindSpeed"].quantile(ub_percentile)
            if lb_percentile >= 0.:
                ws_lb = ws_df["WindSpeed"].quantile(lb_percentile)
            else: # 百分数为负的情况
                ws_min = ws_df["WindSpeed"].quantile(0.)
                ws_symmetric = ws_df["WindSpeed"].quantile(-lb_percentile)
                ws_lb = ws_min - (ws_symmetric - ws_min)
            ws_var = ws_df["WindSpeed"].var()
            if len(ws_df) < small_grid_coef * len(interval_df) / (rs_interval_n * pw_interval_n):
                ws_var = 0.
            rs_grid_series = pd.Series(rs_pw[0], name="rs_grid", index=ws_df.index)
            pw_grid_series = pd.Series(rs_pw[1], name="pw_grid", index=ws_df.index)
            lb_series = pd.Series(ws_lb, name="grid_lb", index=ws_df.index)
            ub_series = pd.Series(ws_ub, name="grid_ub", index=ws_df.index)
            var_series = pd.Series(ws_var, name="grid_var", index=ws_df.index)
            bound_var_df = bound_var_df.append(
                pd.concat([rs_grid_series, pw_grid_series, lb_series, ub_series, var_series], axis=1)
            )
            grid2attr[rs_pw] = {"lb": ws_lb, "ub": ws_ub, "var": ws_var, "n": len(ws_df)}
        inrange_sub_df = pd.concat([inrange_sub_df, bound_var_df], axis=1)

    # 绘制各种图
    draw_figure_for_preprocessing = False
    draw_in_range_only = False
    plot_sub_df = inrange_sub_df
    if draw_in_range_only:
        plot_save_dir = "./figures/inrange/"
    else:
        plot_save_dir = "./figures/alldata/"
    if draw_figure_for_preprocessing:
        # 画数据的三维散点图
        fig = plt.figure()
        fig.set_size_inches(30, 30, 30)
        ax = Axes3D(fig)
        try: # 绘制一次清洗后的数据点，标记每个点所在网格的WindSpeed方差或极差作为颜色
            # ax.set_title("Color stands for grid_var")
            # ax0 = ax.scatter(plot_sub_df["WindSpeed"], plot_sub_df["RotorSpeed"], plot_sub_df["Power"], c=plot_sub_df["grid_var"])
            ax.set_title("Color stands for range")
            ax0 = ax.scatter(plot_sub_df["WindSpeed"], plot_sub_df["RotorSpeed"], plot_sub_df["Power"], c=plot_sub_df["grid_ub"]-plot_sub_df["grid_lb"])
        except: # 绘制一次清洗前的所有数据点
            ax0 = ax.scatter(plot_sub_df["WindSpeed"], plot_sub_df["RotorSpeed"], plot_sub_df["Power"])
        ax.set_xlabel("WindSpeed")
        ax.set_ylabel("RotorSpeed")
        ax.set_zlabel("Power")
        fig.colorbar(ax0)
        plt.savefig(plot_save_dir + str(wind_number) + "_scatter.jpg")
        # plt.show()
        # # 画各个维度上的概率分布函数
        # fig, axs = plt.subplots(1, 3)
        # # fig.set_size_inches(4, 10)
        # fig.suptitle("WindNumber: " + str(wind_number))
        # for i, col in enumerate(plot_sub_df.columns):
        #     axs[i].set_title(col)
        #     axs[i].set_ylim(0, 1)
        #     axs[i].plot(
        #         np.sort(plot_sub_df[col]), 
        #         np.arange(1, len(plot_sub_df)+1) / len(plot_sub_df))
        # plt.savefig(plot_save_dir + str(wind_number) + "_distribution.jpg")
        # # plt.show()
        # # 画各个维度上的概率密度函数
        # fig, axs = plt.subplots(1, 3)
        # # fig.set_size_inches(4, 10)
        # fig.suptitle("WindNumber: " + str(wind_number))
        # for i, col in enumerate(plot_sub_df.columns):
        #     axs[i].set_title(col)
        #     X = np.sort(plot_sub_df[col].values).reshape(-1, 1)
        #     kde_tmp = KernelDensity().fit(X)
        #     y = np.exp(kde_tmp.score_samples(X))
        #     axs[i].plot(X.reshape(-1), y)
        # plt.savefig(plot_save_dir + str(wind_number) + "_density.jpg")
        # # plt.show()
        # # 画维度两两组合的函数关系
        # fig, axs = plt.subplots(1, 3)
        # fig.set_size_inches(40, 20)
        # fig.suptitle("WindNumber: " + str(wind_number))
        # axs[0].set_title("W&P")
        # axs[0].set_xlabel("WindSpeed")
        # axs[0].set_ylabel("Power")
        # ax0 = axs[0].scatter(plot_sub_df["WindSpeed"], plot_sub_df["Power"], c=plot_sub_df["RotorSpeed"])
        # fig.colorbar(ax0, ax=axs[0])
        # axs[1].set_title("W&R")
        # axs[1].set_xlabel("WindSpeed")
        # axs[1].set_ylabel("RotorSpeed")
        # ax1 = axs[1].scatter(plot_sub_df["WindSpeed"], plot_sub_df["RotorSpeed"], c=plot_sub_df["Power"])
        # fig.colorbar(ax1, ax=axs[1])
        # axs[2].set_xlabel("RotorSpeed")
        # axs[2].set_ylabel("Power")
        # axs[2].set_title("R&P")
        # ax2 = axs[2].scatter(plot_sub_df["RotorSpeed"], plot_sub_df["Power"], c=plot_sub_df["WindSpeed"])
        # fig.colorbar(ax2, ax=axs[2])
        # plt.savefig(plot_save_dir + str(wind_number) + "_dim_relation.jpg")
        # plt.show()
        continue

    #####################
    # 异常类别判别逻辑！ #
    #####################
    print("--判别...")
    # 定义label列容器label_series
    label_series = pd.Series(np.zeros(len(sub_df)), index=sub_df.index, name="label", dtype="int32")

    # 清洗判别：首先将参数范围外的点判别为异常
    WASH_PARAM = WASH_BY_PARAM and True
    if WASH_PARAM:
        print("--判别风机参数范围外的点为异常")
        windspeed_not_in_range = ~windspeed_in_range
        label_series[windspeed_not_in_range] = 1 # 将风速在范围外的点视为异常
        rotorspeed_not_in_range = ~rotorspeed_in_range
        label_series[rotorspeed_not_in_range] = 1 # 将转速在范围外的点视为异常

    # 只清洗最底部那一条的离谱点
    WASH_BOTTOM = False
    if WASH_BOTTOM:
        pass

    # 使用网格内grid_var的大小判断网格是否异常，进而去除网格内的异常点，为此先画一下每台风机grid_var的分布验证一下猜想
    GRID_VAR_KDE = GRID_VAR and False
    if GRID_VAR_KDE:
        # 核密度估计
        kde_model_path = "./models/" + str(wind_number) + "_grid_var_kde_bandwidth.pkl"
        if os.path.exists(kde_model_path):
            print("  Using existing kde model: " + kde_model_path)
            with open(kde_model_path, "rb") as f:
                kde = pickle.load(f)
        else:
            print("  Training kde model: " + kde_model_path)
            kde = KernelDensity()
            kde.fit(inrange_sub_df["grid_var"].values.reshape(-1, 1))
            with open(kde_model_path, "wb") as f:
                pickle.dump(kde, f)
        print(" kde scoring...")
        # 画核密度估计结果的三维散点图
        print("  Plotting kde result: " + plot_save_dir + str(wind_number) + "_grid_var_kde_scatter.jpg")
        X = np.linspace(0, inrange_sub_df["grid_var"].max() + 1, 50)
        y = kde.score_samples(X.reshape(-1, 1))
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax0 = ax.plot(X, y)
        ax.set_xlabel("grid_var")
        ax.set_ylabel("prob")
        plt.savefig(plot_save_dir + str(wind_number) + "_grid_var_kde_scatter.jpg")

    # 使用网格内grid_var的大小判断网格是否异常，进而去除网格内的异常点
    GRID_VAR_WASH = GRID and True
    if GRID_VAR_WASH:
        print("--根据grid_var清洗...")
        # 视grid_var的分布为高斯分布，拟合出高斯分布的参数μ和σ
        gm = GaussianMixture()
        gm.fit(inrange_sub_df["grid_var"].values.reshape(-1, 1))
        myu, sigma = float(gm.means_), float(gm.covariances_)
        print("  μ:" , myu, "σ:", sigma)
        # 在那些grid_var过高的网格中，去除windspeed值过高的数据点，直至grid_var小于(μ + kσ)
        for grid_var, grid_df in inrange_sub_df.groupby("grid_var"):
            rs_grid, pw_grid = grid_df["rs_grid"].iloc[0], grid_df["pw_grid"].iloc[0]
            if rs_grid == 0 and pw_grid == 0 and not WASH_BY_PARAM:
                # 对最底部的点做特殊处理，按照邻近grid的ub和lb来截断
                lb_ub = np.zeros(2)
                total_n = 0
                for i, grid in enumerate([(0, 1), (1, 0), (1, 1)]):
                    if grid in grid2attr:
                        attr = grid2attr[grid]
                    else:
                        continue
                    lb_ub += np.array([attr["lb"], attr["ub"]]) * attr["n"]
                    total_n += attr["n"]
                lb_ub /= total_n
                lb, ub = lb_ub
                outlier_index = grid_df[(grid_df["WindSpeed"] > ub) | (grid_df["WindSpeed"] < lb)].index
                label_series[outlier_index] = 1
                continue
            # 若网格grid_var过高，设置clip_value为该网格内windspeed的(1-clip_ratio)百分数，去掉windspeed大于clip_value的点，重新计算新的grid_var
            # 若新的grid_var仍然过高，逐步增大clip_ratio，直至grid_var小于(μ + kσ)
            clip_df_var = grid_var
            clip_ratio = 0.
            var_inc = 0
            while clip_df_var >= (myu + k * sigma):
                clip_ratio += clip_ratio_step
                clip_value = grid_df["WindSpeed"].quantile(1 - clip_ratio)
                clip_df = grid_df[grid_df["WindSpeed"] < clip_value]
                pre_clip_df_var = clip_df_var
                clip_df_var = clip_df["WindSpeed"].var()
                if clip_df_var > pre_clip_df_var:
                    # 如果去除WindSpeed较大的点，方差反而不断增大，表明可能有问题，提前结束该网格的判断
                    var_inc += 1
                    if var_inc >= 2:
                        break
                else:
                    var_inc = 0
            # 被去掉的点视为异常点
            if clip_ratio > 0:
            # if clip_ratio > 0 and clip_ratio < 1.:
                outlier_index = grid_df[grid_df["WindSpeed"] >= clip_value].index
                label_series[outlier_index] = 1
    
    # TODO 使用网格内grid_range与周围网格的grid_range的大小判断网格是否异常，进而去除网格内的异常点
    GRID_RANGE_WASH = GRID and False
    if GRID_RANGE_WASH:
        print(inrange_sub_df)
        exit(0)

    # 机器学习前，归一化各维度
    NORMALIZE = None
    if NORMALIZE == "min-max": # min-max标准化
        scaled_sub_df = inrange_sub_df.copy()
        tmp_df = scaled_sub_df[["WindSpeed", "RotorSpeed", "Power"]]
        tmp_df = (tmp_df - tmp_df.min()) / (tmp_df.max() - tmp_df.min())
    elif NORMALIZE == "Z-Score": # Z-Score标准化
        scaled_sub_df = inrange_sub_df.copy()
        tmp_df = scaled_sub_df[["WindSpeed", "RotorSpeed", "Power"]]
        tmp_df = (tmp_df - tmp_df.mean()) / tmp_df.std()
    else:
        scaled_sub_df = inrange_sub_df
    
    # TODO 尝试：Kmeans聚类


    # 尝试：核密度估计
    KDE = False
    if KDE:
        # 核密度估计参数
        bandwidth = 1.0
        algorithm = "auto"
        kernel = "gaussian"
        clip_percentile = 0.05
        # 核密度估计
        kde_model_path = "./models/" + str(wind_number) + "_kde_bandwidth" + str(bandwidth) + ".pkl"
        if os.path.exists(kde_model_path):
            print("  Using existing kde model: " + kde_model_path)
            with open(kde_model_path, "rb") as f:
                kde = pickle.load(f)
        else:
            print("  Training kde model: " + kde_model_path)
            kde = KernelDensity()
            kde.fit(scaled_sub_df["WindSpeed", "RotorSpeed", "Power"])
            with open(kde_model_path, "wb") as f:
                pickle.dump(kde, f)
        print(" kde scoring...")
        scores = kde.score_samples(sub_df.values)
        clip_prob = np.sort(scores)[int(clip_percentile*len(scores))]
        label_series[scores < clip_prob] = 1
        # 画核密度估计结果的三维散点图
        print("  Plotting kde result: " + plot_save_dir + str(wind_number) + "_kde_scatter.jpg")
        fig = plt.figure()
        fig.set_size_inches(30, 30, 30)
        ax = Axes3D(fig)
        ax.set_title("Color stands for power")
        plot_scores = scores[windspeed_in_range & rotorspeed_in_range] if draw_in_range_only else scores
        ax0 = ax.scatter(plot_sub_df["WindSpeed"], plot_sub_df["RotorSpeed"], plot_sub_df["Power"], c=plot_scores)
        ax.set_xlabel("WindSpeed")
        ax.set_ylabel("RotorSpeed")
        ax.set_zlabel("Power")
        fig.colorbar(ax0)
        plt.savefig(plot_save_dir + str(wind_number) + "_kde_scatter.jpg")

    # 线性回归
    # 可以试一下改进线性回归
    LR = False
    if LR:
        columns = list(inrange_sub_df.columns)
        rs_col_id= columns.index("RotorSpeed")
        pw_col_id= columns.index("Power")
        ws_col_id= columns.index("WindSpeed")
        # 对RotorSpeed-WindSpeed线性回归
        lr_rs = LinearRegression()
        lr_rs.fit(scaled_sub_df[:, rs_col_id].reshape(-1, 1), scaled_sub_df[:, ws_col_id])
        print(lr_rs.coef_, lr_rs.intercept_)

    # TODO 直接建立线性模型
    LINEAR = False
    if LINEAR:
        pass


    # 构建输出DataFrame，并stack到最终结果output_df上
    output_sub_df = pd.concat([
        # raw_sub_df["WindNumber"], 
        # raw_sub_df["Time"], 
        sub_df, 
        label_series], 
        axis=1) #构建空的预测结果的DataFrame
    print(output_sub_df["label"].value_counts())
    output_df = pd.concat([output_df, output_sub_df], axis=0, sort=True)

    # 画结果的三维散点图
    PLOT_RESULT = True
    if PLOT_RESULT:
        fig = plt.figure()
        fig.set_size_inches(30, 30, 30)
        ax = Axes3D(fig)
        ax.set_title("Color stands for label")
        ax0 = ax.scatter(output_sub_df["WindSpeed"], output_sub_df["RotorSpeed"], output_sub_df["Power"], c=output_sub_df["label"])
        ax.set_xlabel("WindSpeed")
        ax.set_ylabel("RotorSpeed")
        ax.set_zlabel("Power")
        fig.colorbar(ax0)
        plt.savefig(plot_save_dir + str(wind_number) + "_results_scatter.jpg")
        # plt.show()


# 统计结果
print("--结果统计：")
print(output_df["label"].value_counts())

# 输出结果至文件
output_df = pd.concat([
    output_df["WindNumber"], 
    output_df["Time"], 
    output_df["label"] 
    ], axis=1)
output_df.to_csv("./results/result.csv", index=False)
