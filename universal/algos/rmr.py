import numpy as np
import pandas as pd

from universal import tools
from universal.algo import Algo
from universal.algos.olmar import OLMAR


def norm(x):
    if isinstance(x, pd.Series):
        axis = 0
    else:
        axis = 1
    return np.sqrt((x ** 2).sum(axis=axis))


class RMR(OLMAR):
    """Robust Median Reversion. Strategy exploiting mean-reversion by robust
    L1-median estimator. Practically the same as OLMAR.

    Reference:
        Dingjiang Huang, Junlong Zhou, Bin Li, Steven C.H. Hoi, Shuigeng Zhou
        Robust Median Reversion Strategy for On-Line Portfolio Selection, 2013.
        http://ijcai.org/papers13/Papers/IJCAI13-296.pdf
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10.0, tau=0.001):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        :param tau: Precision for finding median. Recommended value is around 0.001. Strongly
                    affects algo speed.
        """
        super().__init__(window, eps)
        self.tau = tau

    def predict(self, x, history):
        """find L1 median to historical prices"""
        y = history.mean()
        y_last = None
        while y_last is None or norm(y - y_last) / norm(y_last) > self.tau:
            y_last = y
            d = norm(history - y)
            y = history.div(d, axis=0).sum() / (1.0 / d).sum()
        return y / x


if __name__ == "__main__":
    data0 = tools.dataset("msci")  # 读取数据
    result = tools.quickrun(RMR(),data0)

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
