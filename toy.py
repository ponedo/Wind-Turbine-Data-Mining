# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.cluster import DBSCAN

# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                       noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)

# X = np.concatenate((X1, X2))
# y_pred = DBSCAN(eps = 0.1).fit_predict(X)
# # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# # plt.show()
# print(X)
# print(y_pred)
# print(np.unique(y_pred))
# print(np.argwhere(y_pred == 1))


import pandas as pd
raw_df = pd.DataFrame([[1,2,3],[3,2,13],[2,4,3],[23,4,12],[6,5,34]])
df = raw_df
print(df)
df.loc[0, 1] = 100
print(df)
df = df[df[2]==3]
print(df)
for value1, sub_df in df.groupby(1):
    if value1 == 100:
        df.loc[sub_df.index, 1] = 1
print(raw_df)
print(df)
