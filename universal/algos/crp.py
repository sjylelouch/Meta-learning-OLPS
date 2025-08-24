import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from universal import tools
from universal.algo import Algo


class CRP(Algo):
    """Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        b表示不断重新平衡的投资组合权重。默认值为统一。
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super().__init__() #调用父类的init方法
        self.b = np.array(b) if b is not None else None

    def step(self, x, last_b, history):
        # init b to default if necessary
        #如果b=none使用uniform方法，x表示股票列表
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b

    def weights(self, X):
        if self.b is None:
            b = X * 0 + 1
            b.loc[:, "CASH"] = 0
            #标准化处理，每一行的和=1
            b = b.div(b.sum(axis=1), axis=0)
            return b
        #mdim表示数据的维度（1维、2维、3维），X表示天数
        elif self.b.ndim == 1:
            return np.repeat([self.b], X.shape[0], axis=0)#shape[0]表示列数
        else:
            return self.b

    @classmethod
    def plot_crps(cls, data, show_3d=False):
        """Plot performance graph for all CRPs (Constant Rebalanced Portfolios).
        :param data: Stock prices.
        :param show_3d: Show CRPs on a 3-simplex, works only for 3 assets.
        """

        def _crp(data):
            B = list(tools.simplex_mesh(2, 100))
            crps = CRP.run_combination(data, b=B)
            x = [b[0] for b in B]
            y = [c.total_wealth for c in crps]
            return x, y

        # init
        import ternary

        data = data.dropna(how="any")
        data = data / data.iloc[0]
        dim = data.shape[1]

        # plot prices
        if dim == 2 and not show_3d:
            fig, axes = plt.subplots(ncols=2, sharey=True)
            data.plot(ax=axes[0], logy=True)
        else:
            data.plot(logy=False)

        if show_3d:
            assert dim == 3, "3D plot works for exactly 3 assets."
            plt.figure()
            fun = lambda b: CRP(b).run(data).total_wealth
            ternary.plot_heatmap(fun, steps=20, boundary=True)

        elif dim == 2:
            x, y = _crp(data)
            s = pd.Series(y, index=x)
            s.plot(ax=axes[1], logy=True)
            plt.title("CRP performance")
            plt.xlabel("weight of {}".format(data.columns[0]))

        elif dim > 2:
            fig, axes = plt.subplots(ncols=dim - 1, nrows=dim - 1)
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    x, y = _crp(data[[i, j]])
                    ax = axes[i][j - 1]
                    ax.plot(x, y)
                    ax.set_title("{} & {}".format(data.columns[i], data.columns[j]))
                    ax.set_xlabel("weights of {}".format(data.columns[i]))


if __name__ == "__main__":
    data = tools.dataset("djia")  # 读取数据
    result = tools.quickrun(CRP(),data)
    #print(result.information)
    #
    # weight = np.array(result.weights)[:, 0:28]
    # money = result.total_wealth
    # for i in range(1,np.shape(weight)[0]):
    #     w = np.multiply(weight[i-1], np.array(data)[i-1, :])/np.sum(np.multiply(weight[i-1], np.array(data)[i-1, :]))
    #     money = money*(1-0.01*np.sum(np.abs(weight[i]-w)))
    #     print(money)