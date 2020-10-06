####################################################################################################################
# 5. 针对10号风机的修改
#   For each wind turbine:
#     Divide wind speed values into a number of equal intervals.
#     The DBSCAN clustering method is applied to the wind power dataset in each wind speed interval.
#     The topmost cluster with largest average power value is the normal data, while other clusters are eliminated.
####################################################################################################################
print("Specialize for each wind turbine 10...")
df = raw_df.loc[raw_df_index]
df = df[df["label"] == 0]
for wind_number, sub_df in df.groupby("WindNumber"):
    if DEBUG and not wind_number == DEBUG_WIND_NUMBER:
        continue
    if not (wind_number == 10):
        continue
    x, y = sub_df["WindSpeed"], sub_df["RotorSpeed"]
    y_fit = np.polyfit(x, y, 2)  # 二次多项式拟合
    y_show = np.poly1d(y_fit)
    # print(y_show)
    # x_plot = np.arange(2.5, 10, 0.01)
    # y_plot = y_show.coef[0] * (
    #     (x_plot + 0.8)**2) + y_show.coef[1] * (x_plot + 0.8) + y_show.coef[2]
    sub_df['diff'] = sub_df['RotorSpeed'] - (
        y_show.coef[0] * ((sub_df["WindSpeed"] + 0.8)**2) + y_show.coef[1] *
        (sub_df["WindSpeed"] + 0.8) + y_show.coef[2])
    outlier_manifold_index = sub_df[(sub_df["diff"] > 0)
                                    & (sub_df["WindSpeed"] > 0) &
                                    (sub_df["WindSpeed"] <= 5)].index
    raw_df.loc[outlier_manifold_index, "label"] = 1
print(raw_df.loc[raw_df_index, "label"].value_counts())