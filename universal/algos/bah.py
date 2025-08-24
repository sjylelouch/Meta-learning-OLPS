import numpy as np
from universal import tools
from universal.algo import Algo
import pandas as pd


class BAH(Algo):
    """Buy and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever."""

    PRICE_TYPE = "raw"

    def __init__(self, b=None):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super().__init__()
        self.b = b

    def weights(self, S):
        """Weights function optimized for performance."""
        if self.b is None:
            b = np.array([0 if s == "CASH" else 1 for s in S.columns])
            b = b / b.sum()
        else:
            b = self.b
        print(b)
        # weights are proportional to price times initial weights
        w = S.shift(1) * b

        # normalize
        w = w.div(w.sum(axis=1), axis=0)

        w.iloc[0] = 1.0 / S.shape[1]

        return w


if __name__ == "__main__":
    # tools.quickrun(BAH())
    data0 = tools.dataset("msci")  # 读取数据
    n = np.shape(data0)[0]-int(np.shape(data0)[0]*0.8)-11
    data1 = data0[-n-1:-1]
    data1 = data1.reset_index(drop=True)

    # data0 = []
    # path = r"C:\Users\shenjiayu\Desktop\元学习在线投资组合\universal-portfolios-master\universal-portfolios-master\universal\data\stock0.csv"
    # data = pd.read_csv(path)
    #设定投资组合策略
    result = tools.quickrun(BAH(), data1)
    equity = np.array(result.equity)
    np.savetxt("msci.csv", equity, delimiter=',', fmt='%.3f')
    # data = np.zeros([np.shape(data0)[0] - 1, np.shape(data0)[1]])
    # for i in range(np.shape(data)[0]):
    #     data[i, :] = np.divide(np.array(data0)[i+1, :], np.array(data0)[i, :])
    # # excel_path = r"C:\Users\shenjiayu\Desktop\BAH_weights1.xlsx"
    # # result.weights.to_excel(excel_path, index=False)
    #
    # money = result.total_wealth
    # weight = np.array(result.weights)
    # for i in range(1,np.shape(weight)[0]):
    #     w = np.multiply(weight[i-1], np.array(data)[i-1, :])/np.sum(np.multiply(weight[i-1], np.array(data)[i-1, :]))
    #     money = money*(1-0.0005*np.sum(np.abs(weight[i]-w)))
    #     print(money)

    S = result.total_wealth
    S ** (250 / 1120) - 1