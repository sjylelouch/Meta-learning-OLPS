import numpy as np
import pandas as pd

from universal import tools
from universal.algo import Algo


class OLMAR(Algo):
    """On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super().__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError("window parameter must be >=3")
        if eps < 1:
            raise ValueError("epsilon parameter must be >=1")

        self.window = window
        self.eps = eps

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window :])
        b = self.update(last_b, x_pred, self.eps)
        return b

    def predict(self, x, history):
        """Predict returns on next day."""
        return (history / x).mean()

    def update(self, b, x_pred, eps):
        """Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights."""
        x_pred_mean = np.mean(x_pred)
        lam = max(
            0.0, (eps - np.dot(b, x_pred)) / np.linalg.norm(x_pred - x_pred_mean) ** 2
        )

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x_pred - x_pred_mean)

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == "__main__":
    data0 = tools.dataset("djia")  # 读取数据
    result = tools.quickrun(OLMAR(),data0)
    #
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
