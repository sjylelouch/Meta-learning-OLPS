import numpy as np
import pandas as pd

from universal import tools
from universal.algos.crp import CRP


class BestMarkowitz(CRP):
    """Optimal Markowitz portfolio constructed in hindsight.

    Reference:
        https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = False

    def __init__(self, global_sharpe=None, sharpe=None, **kwargs):
        self.global_sharpe = global_sharpe
        self.sharpe = sharpe
        self.opt_markowitz_kwargs = kwargs

    def weights(self, X):
        """Find optimal markowitz weights."""
        # update frequency
        freq = tools.freq(X.index)#年化

        R = X - 1

        # calculate mean and covariance matrix and annualize them
        sigma = R.cov() * freq

        if self.sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)), X.columns) * pd.Series(
                self.sharpe
            ).reindex(X.columns)
        elif self.global_sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)) * self.global_sharpe, X.columns)
        else:
            mu = R.mean() * freq

        self.b = tools.opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

        return super().weights(R)


if __name__ == "__main__":
    data0 = tools.dataset("tse")  # 读取数据
    result = tools.quickrun(BestMarkowitz(),data0)

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
    weight = np.array(result.weights)[:, 0:np.shape(data)[1]]