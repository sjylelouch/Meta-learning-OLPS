import numpy as np
import pandas as pd
import scipy.stats
from numpy import diag, log, sqrt, trace
from numpy.linalg import inv

from universal import tools
from universal.algo import Algo


class CWMR(Algo):
    """Confidence weighted mean reversion.

    Reference:
        B. Li, S. C. H. Hoi, P.L. Zhao, and V. Gopalkrishnan.
        Confidence weighted mean reversion strategy for online portfolio selection, 2013.
        http://jmlr.org/proceedings/papers/v15/li11b/li11b.pdf
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, eps=-0.5, confidence=0.95):
        """
        :param eps: Mean reversion threshold (expected return on current day must be lower
                    than this threshold). Recommended value is -0.5.
        :param confidence: Confidence parameter for profitable mean reversion portfolio.
                    Recommended value is 0.95.
        """
        super().__init__()

        # input check
        if not (0 <= confidence <= 1):
            raise ValueError("confidence must be from interval [0,1]")

        self.eps = eps
        self.theta = scipy.stats.norm.ppf(confidence)#标准正态分布累计概率为0.95时的置信度

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        m = X.shape[1]
        self.sigma = np.matrix(np.eye(m) / m ** 2)

    def step(self, x, last_b, history):
        # initialize
        m = len(x)
        mu = np.matrix(last_b).T
        sigma = self.sigma
        theta = self.theta
        eps = self.eps
        x = np.matrix(x).T  # matrices are easier to manipulate

        # 4. Calculate the following variables
        M = mu.T * x
        V = x.T * sigma * x
        x_upper = sum(diag(sigma) * x) / trace(sigma)

        # 5. Update the portfolio distribution
        mu, sigma = self.update(x, x_upper, mu, sigma, M, V, theta, eps)

        # 6. Normalize mu and sigma
        mu = tools.simplex_proj(mu)
        sigma = sigma / (m ** 2 * trace(sigma))
        """
        sigma(sigma < 1e-4*eye(m)) = 1e-4;
        """
        self.sigma = sigma
        return mu

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        # lambda from equation 7
        foo = (
            V - x_upper * x.T * np.sum(sigma, axis=1)
        ) / M ** 2 + V * theta ** 2 / 2.0
        a = foo ** 2 - V ** 2 * theta ** 4 / 4
        b = 2 * (eps - log(M)) * foo
        c = (eps - log(M)) ** 2 - V * theta ** 2

        a, b, c = a[0, 0], b[0, 0], c[0, 0]

        lam = max(
            0,
            (-b + sqrt(b ** 2 - 4 * a * c)) / (2.0 * a),
            (-b - sqrt(b ** 2 - 4 * a * c)) / (2.0 * a),
        )
        # bound it due to numerical problems
        lam = min(lam, 1e7)

        # update mu and sigma
        U_sqroot = 0.5 * (
            -lam * theta * V + sqrt(lam ** 2 * theta ** 2 * V ** 2 + 4 * V)
        )
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + theta * lam / U_sqroot * diag(x) ** 2)
        """
        tmp_sigma = inv(inv(sigma) + theta*lam/U_sqroot*diag(xt)^2);
        % Don't update sigma if results are badly scaled.
        if all(~isnan(tmp_sigma(:)) & ~isinf(tmp_sigma(:)))
            sigma = tmp_sigma;
        end
        """
        return mu, sigma


class CWMR_VAR(CWMR):
    """First variant of a CWMR outlined in original article. It is
    only approximation to the posted problem."""

    def update(self, x, x_upper, mu, sigma, M, V, theta, eps):
        # lambda from equation 7
        foo = (V - x_upper * x.T * np.sum(sigma, axis=1)) / M ** 2
        a = 2 * theta * V * foo
        b = foo + 2 * theta * V * (eps - log(M))
        c = eps - log(M) - theta * V

        a, b, c = a[0, 0], b[0, 0], c[0, 0]

        lam = max(
            0,
            (-b + sqrt(b ** 2 - 4 * a * c)) / (2.0 * a),
            (-b - sqrt(b ** 2 - 4 * a * c)) / (2.0 * a),
        )
        # bound it due to numerical problems
        lam = min(lam, 1e7)

        # update mu and sigma
        mu = mu - lam * sigma * (x - x_upper) / M
        sigma = inv(inv(sigma) + 2 * lam * theta * diag(x) ** 2)
        """
        tmp_sigma = inv(inv(sigma) + theta*lam/U_sqroot*diag(xt)^2);
        % Don't update sigma if results are badly scaled.
        if all(~isnan(tmp_sigma(:)) & ~isinf(tmp_sigma(:)))
            sigma = tmp_sigma;
        end
        """
        return mu, sigma


# use case
if __name__ == "__main__":
    data0 = tools.dataset("tse")  # 读取数据
    result = tools.quickrun(CWMR(),data0)

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
