import numpy as np
from cvxopt import matrix, solvers

from universal import tools
from universal.algo import Algo
import pandas as pd

solvers.options["show_progress"] = False


class ONS(Algo):
    """
    Online newton step algorithm.

    Reference:
        A.Agarwal, E.Hazan, S.Kale, R.E.Schapire.
        Algorithms for Portfolio Management based on the Newton Method, 2006.
        http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_AgarwalHKS06.pdf
    """

    REPLACE_MISSING = True

    def __init__(self, delta=0.125, beta=1.0, eta=0.0):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super().__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        m = X.shape[1]
        self.A = np.mat(np.eye(m))
        self.b = np.mat(np.zeros(m)).T

    def step(self, r, p, history):
        # calculate gradient
        grad = np.mat(r / np.dot(p, r)).T
        # update A
        self.A += grad * grad.T
        # update b
        self.b += (1 + 1.0 / self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M):
        """Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
        m = M.shape[0]

        P = matrix(2 * M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A = matrix(np.ones((1, m)))
        b = matrix(1.0)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol["x"])


if __name__ == "__main__":
    data0 = tools.dataset("nyse_o")  # 读取数据
    result = tools.quickrun(ONS(), data0)
    #
    # path = r"C:\Users\shenjiayu\Desktop\元学习在线投资组合\universal-portfolios-master\universal-portfolios-master\universal\data\stock0.csv"
    # data = pd.read_csv(path)
    # 设定投资组合策略
    # excel_path = r"C:\Users\shenjiayu\Desktop\BAH_weights1.xlsx"
    # result.weights.to_excel(excel_path, index=False)
    data = np.zeros([np.shape(data0)[0] - 1, np.shape(data0)[1]])
    for i in range(np.shape(data)[0]):
        data[i, :] = np.divide(np.array(data0)[i + 1, :], np.array(data0)[i, :])
    # excel_path = r"C:\Users\shenjiayu\Desktop\BAH_weights1.xlsx"
    # result.weights.to_excel(excel_path, index=False)

    money = result.total_wealth
    weight = np.array(result.weights)
    for i in range(1, np.shape(weight)[0]):
        w = np.multiply(weight[i - 1], np.array(data)[i - 1, :]) / np.sum(
            np.multiply(weight[i - 1], np.array(data)[i - 1, :]))
        money = money * (1 - 0.0005 * np.sum(np.abs(weight[i] - w)))
        print(money)
    weight = np.array(result.weights)[:,0:np.shape(data)[1]]
    S=1
    for i in range(1, np.shape(weight)[0]):
        w = np.multiply(weight[i - 1], np.array(data)[i - 1, :]) / np.sum(
            np.multiply(weight[i - 1], np.array(data)[i - 1, :]))
        S=S * np.sum(np.multiply(weight[i - 1], np.array(data)[i - 1, :]))* (1 - 0.01 * np.sum(np.abs(weight[i] - w)))
        # S = S * np.sum(np.multiply(weight[i - 1], np.array(data)[i - 1, :]))