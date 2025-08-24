#导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

#导入策略库
from universal import tools
from universal.algos import CRP
from universal.algos import EG
from universal.algos import PAMR
from universal.algos import BNN
from universal.algos import ONS
from universal.algos import UP
from universal.algos import BAH
from universal.algos import DynamicCRP
from universal.algos import Anticor
from universal.algos import CWMR
from universal.algos import BestSoFar
from universal.algos import CORN
from universal.algos import Kelly
from universal.algos import OLMAR
from universal.algos import RMR
from universal.algos import WMAMR
from sklearn.cluster import KMeans

path = "meta-learning_LMPS/data/djia.csv"
data = pd.read_csv(path)
result0 = tools.quickrun(BAH(), data)
result1 = tools.quickrun(UP(), data)
result2 = tools.quickrun(PAMR(), data)
result3 = tools.quickrun(Anticor(), data)
result4 = tools.quickrun(CWMR(), data)
result5 = tools.quickrun(BestSoFar(), data)
result6 = tools.quickrun(CORN(), data)
result7 = tools.quickrun(BNN(), data)
result8 = tools.quickrun(OLMAR(), data)
result9 = tools.quickrun(RMR(), data)
result10 = tools.quickrun(WMAMR(), data)
result11 = tools.quickrun(CRP(), data)
result12 = tools.quickrun(EG(), data)

#
d = np.array(data).shape[0]*3/4
S_0 = np.array(result0.equity)[:int(d)]
S_1 = np.array(result1.equity)[:int(d)]
S_2 = np.array(result2.equity)[:int(d)]
S_3 = np.array(result3.equity)[:int(d)]
S_4 = np.array(result4.equity)[:int(d)]
S_5 = np.array(result5.equity)[:int(d)]
S_6 = np.array(result0.equity)[:int(d)]
S_7 = np.array(result7.equity)[:int(d)]
S_8 = np.array(result8.equity)[:int(d)]
S_9 = np.array(result9.equity)[:int(d)]
S_10 = np.array(result10.equity)[:int(d)]
S_11 = np.array(result11.equity)[:int(d)]
S_12 = np.array(result12.equity)[:int(d)]
# S_13 = np.array(result13.equity)[:int(d)]



vectors = [S_0,S_1,S_3,S_4,S_6,S_7,S_8,S_9,S_10,S_11,S_12]
# vectors = [S_0,S_1,S_2,S_3,S_4,S_5,S_6,S_7,S_8,S_9,S_10,S_11,S_12,S_13]  # 14个向量的列表

#对数据做标准化处理
row_means = np.mean(vectors, axis=1)
row_stds = np.std(vectors, axis=1)
normalized_data = (vectors - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]

k = 4  # 聚类数量
kmeans = KMeans(n_clusters=k, random_state=42).fit(normalized_data)  # 使用KMeans进行聚类

# 获取每个向量所属的类别标签
labels = kmeans.labels_

# 初始化每个类别的向量索引列表
cluster_indices = [[] for _ in range(k)]

# 将向量索引分配到对应的类别列表中
for i, label in enumerate(labels):
    cluster_indices[label].append(i)

# 输出每个类别中向量的索引
for i, indices in enumerate(cluster_indices):
    print(f"Class {i+1} indices: {indices}")

# 计算每一类累计收益率最大的策略
vectors = np.array(vectors)
choose = np.zeros([k])
shouyi = np.zeros([k])
for i in range(k):
    m = 0
    for j in range(len(cluster_indices[i])):
        if vectors[cluster_indices[i][j], -1] > m:
            choose[i] = cluster_indices[i][j]
            shouyi[i] = vectors[cluster_indices[i][j], -1]
            m = shouyi[i]

# plt.plot(np.log(S_0),label='BAH')
# plt.plot(np.log(S_1),label='UP')
# plt.plot(np.log(S_2),label='DynamicCRP')
# plt.plot(np.log(S_3),label='Anticor')
# plt.plot(np.log(S_4),label='CWMR')
# plt.plot(np.log(S_5),label='BestSoFar')
# plt.plot(np.log(S_6),label='CORN')
# plt.plot(np.log(S_7),label='BNN')
# plt.plot(np.log(S_8),label='OLMAR')
# plt.plot(np.log(S_9),label='RMR')
# plt.plot(np.log(S_10),label='WMAMR')
# plt.plot(np.log(S_11),label='CRP')
# plt.plot(np.log(S_12),label='EG')
# plt.plot(np.log(S_13),label='PAMR')
# plt.legend()
# plt.title('nyse_o dataset')
# plt.show()

print(choose)
print(shouyi)