import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing


def dataset(name):
    """Return sample dataset from /data directory."""
    filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "data", name + ".csv"
    )
    return pd.read_csv(filename)

#指数平滑优化问题目标函数
def objective(theta, data):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit(smoothing_level=theta)
    return model_fit.aic

def optimize_smoothing(data):
    result = minimize(objective, x0=0.5, args=(data,))
    best_theta = result.x[0]
    return best_theta


def predict(R, W):
    """
    params R: 历史股票收益序列——t(天数）行，m（股票数量）列
    params W：窗口长度
    params theta:下降参数
    return 0or1：0表示熊市，1表示牛市
    """
    # 计算窗口数目
    d = int(np.shape(R)[0]/W)
    # 构造均值miu和标准差nu向量
    miu = np.zeros(d+1)
    nu = np.zeros(d+1)
    for i in range(d+1):
        if i==d:
            miu[i] = np.mean(R[W*d:])
            nu[i] = np.std(R[W*d:], ddof=1)
        else:
            miu[i] = np.mean(R[W*i:W*i+W])
            nu[i] = np.std(R[W*i:W*i+W], ddof=1)

    # 拟合miu和nu之间的线性关系
    alpha = (sum(miu*nu)-(d+1)*(np.mean(miu)*np.mean(nu)))/(sum(nu*nu)-(d+1)*np.mean(nu)*np.mean(nu))
    beta = np.mean(miu)-alpha*np.mean(nu)

    # 构造R1\R0向量
    R1=np.array([])
    R0=np.array([])
    for i in range(d):
        if miu[i] > alpha*nu[i]+beta:
            R1 = np.concatenate([R1, R[W*i:W*i+W]])
        else:
            R0 = np.concatenate([R0, R[W*i:W*i+W]])
    if miu[d] > alpha*nu[i]+beta:
        R1 = np.concatenate([R1, R[W*d:]])
    else:
        R0 = np.concatenate([R0, R[W*d:]])

    # 判断牛熊市并且预测第t天收益率
    r = 1
    if miu[d-1] > alpha*nu[d-1]+beta:
        #利用优化获得最佳的theta
        # theta = optimize_smoothing(R1[len(R1)-W:len(R1)])
        theta = 0.4
        # print("theta=", theta)
        for i in range(W+1):
            # r = theta+(1-theta)*r/R1[len(R1)-W-1+i]
            r = theta*R1[len(R1)-W-1+i]+(1-theta)*r
        return 1, r
    else:
        # theta = optimize_smoothing(R0[len(R0)-W:len(R0)])
        theta = 0.4
        # print("theta=",theta)
        for i in range(W+1):
            # r = theta+(1-theta)*r/R0[len(R0)-W-1+i]
            r = theta*R0[len(R0)-W-1+i]+(1-theta)*r
        return 0, r

def get_regimeandr(H, W):
    """
        params H:历史股票数据
        return  regime :该时期每只股票市场环境
        return  r：下一时期每只股票收益率预测
    """
    regime = []  # 用于储存每支股票的涨跌
    r = []  # 用于储存每只股票第t日的
    for col in H.columns:
        x = predict(H[col], W)
        regime.append(x[0])
        r.append(x[1])
    return regime, r


if __name__ == "__main__":
    #数据读取与参数设置
    data0 = dataset("nyse_o") # 读取数据
    data = np.zeros([np.shape(data0)[0]-1, np.shape(data0)[1]])
    for i in range(np.shape(data)[0]):
        data[i, :] = np.divide(np.array(data0)[i+1, :], np.array(data0)[i, :])
    data = pd.DataFrame(data)
    W = 30 # 设置窗口长度

    error = 0
    num_equal = 0


    for i in range(1, np.shape(data)[0]):
    # for i in range(1, 40):
        print("i=",i)
        if i > W or i == W:
            # data1 = data0[0:i]
            data1 = data[0:i]
            res = get_regimeandr(data1, W)
            # ra = np.divide(np.array(res[1]), np.array(data0[i-1:i]))
            ra = res[1]
            # print("ra=",ra)
            # print("data=", np.array(data)[i-1, :])
            error = error + np.linalg.norm(ra-np.array(data)[i-1, :])*np.linalg.norm(ra-np.array(data)[i-1, :])
            idx1 = np.argsort(ra)[-3:]
            idx2 = np.argsort(np.array(data)[i - 1, :])[-3:]
            equal_elements = idx1 == idx2
            num_equal = num_equal + np.count_nonzero(equal_elements)

    print("error=", error)
    print("num_equal", num_equal/3/(np.shape(data0)[0]-29))
