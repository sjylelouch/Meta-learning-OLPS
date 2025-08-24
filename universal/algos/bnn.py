# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from universal import tools
from universal.algo import Algo


class BNN(Algo):
    """Nearest neighbor based strategy. It tries to find similar sequences of price in history and
    then maximize objective function (that is profit) on the days following them.

    Reference:
        L. Gyorfi, G. Lugosi, and F. Udina. Nonparametric kernel based sequential
        investment strategies. Mathematical Finance 16 (2006) 337–357.
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, k=5, l=10):
        """
        :param k: Sequence length.
        :param l: Number of nearest neighbors.
        """

        super().__init__(min_history=k + l - 1)

        self.k = k
        self.l = l

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # find indices of nearest neighbors throughout history
        ixs = self.find_nn(history, self.k, self.l)

        # get returns from the days following NNs
        J = history.iloc[[history.index.get_loc(i) + 1 for i in ixs]]

        # get best weights
        return tools.bcrp_weights(J)

    def find_nn(self, H, k, l):
        """Note that nearest neighbors are calculated in a different (more efficient) way than shown
        in the article.

        param H: history
        """
        # calculate distance from current sequence to every other point
        D = H * 0
        for i in range(1, k + 1):
            D += (H.shift(i - 1) - H.iloc[-i]) ** 2
        D = D.sum(1).iloc[:-1]

        # sort and find nearest neighbors
        D = D.sort_values()
        return D.index[:l]


if __name__ == "__main__":
    data0 = tools.dataset("nyse_o")  # 读取数据
    result = tools.quickrun(BNN(), data0)

    # data = np.zeros([np.shape(data0)[0] - 1, np.shape(data0)[1]])
    # for i in range(np.shape(data)[0]):
    #     data[i, :] = np.divide(np.array(data0)[i + 1, :], np.array(data0)[i, :])
    # # excel_path = r"C:\Users\shenjiayu\Desktop\BAH_weights1.xlsx"
    # # result.weights.to_excel(excel_path, index=False)
    #
    # money = result.total_wealth
    # weight = np.array(result.weights)
    # for i in range(1, np.shape(weight)[0]):
    #     w = np.multiply(weight[i - 1], np.array(data)[i - 1, :]) / np.sum(
    #         np.multiply(weight[i - 1], np.array(data)[i - 1, :]))
    #     money = money * (1 - 0.0005 * np.sum(np.abs(weight[i] - w)))
    #     print(money)
