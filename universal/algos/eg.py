# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from universal import tools
from universal.algo import Algo

class EG(Algo):
    """Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
    """

    def __init__(self, eta=0.05):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super().__init__()
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        b = last_b * np.exp(self.eta * x / sum(x * last_b))
        return b / sum(b)


if __name__ == "__main__":
    data0 = tools.dataset("nyse_o")
    result = tools.quickrun(EG(eta=0.5), data0)
    #
    # data = np.zeros([np.shape(data0)[0] - 1, np.shape(data0)[1]])
    # for i in range(np.shape(data)[0]):
    #     data[i, :] = np.divide(np.array(data0)[i + 1, :], np.array(data0)[i, :])
    # money = result.total_wealth
    # weight = np.array(result.weights)
    # for i in range(1, np.shape(weight)[0]):
    #     w = np.multiply(weight[i - 1], np.array(data)[i - 1, :]) / np.sum(
    #         np.multiply(weight[i - 1], np.array(data)[i - 1, :]))
    #     money = money * (1 - 0.0005 * np.sum(np.abs(weight[i] - w)))
    #     print(money)
    # weight = np.array(result.weights)[:,0:np.shape(data)[1]]
    # S=1
    # for i in range(1, np.shape(weight)[0]):
    #     w = np.multiply(weight[i - 1], np.array(data)[i - 1, :]) / np.sum(
    #         np.multiply(weight[i - 1], np.array(data)[i - 1, :]))
    #     S=S * np.sum(np.multiply(weight[i - 1], np.array(data)[i - 1, :]))* (1 - 0.01 * np.sum(np.abs(weight[i] - w)))