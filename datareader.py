import random
import numpy as np
import pandas as pd
import random
from numpy.lib.stride_tricks import as_strided as stride
from config import args


# from PyEMD import EEMD

# 此代码用于处理数据集，数据集的格式为[特征：策略1234一个window的历史收益数据，标签为：下一日四种策略的权重向量/下一日的标签]
class mamldata:
    output_folder = "meta-learning_LMPS/result"  # folder for saving the results
    x_seeds = []
    y_seeds = []
    pw_seeds = []
    rp_seeds = []

    def __init__(self, p, portfolio_weight, portfolio_equity, re_price, batch_size=3, divide=1 / 6, nIMFs=4, train=1):
        """
                :param p: 窗口大小
                :param batch_size: 有多少个不同的任务
                :param k_shot: 一个类中有几个图片用于Inner looper的训练
                :param q_query: 一个类中有几个图片用于Outer looper的训练
                :param divide: 划分训练集比例
                :param nIMFs: 经VMD分解后的IMF个数
        """
        self.p = p
        self.meta_batch_size = batch_size
        self.port_pw = portfolio_weight #策略数量*天数*各股权重
        self.port_rp = re_price  # 策略数量*天数*各股权重
        self.port_eq = portfolio_equity # 每一列表示不同的策略
        self.divide = divide
        self.nIMFs = nIMFs
        self.train = train
        self.steps = 10
        self.nport = 4
        self.nstock = portfolio_weight.shape[2]
        self.data = self.generate_traintest()

    def readdata(self):
        return self.get_return()


    def get_return(self):
        # 这里se_in是原始股票价格数据（dataframe格式）
        result_0 = np.array(self.port_eq[0, :])
        result_1 = np.array(self.port_eq[1, :])
        result_2 = np.array(self.port_eq[2, :])
        result_3 = np.array(self.port_eq[3, :])

        S_0 = np.divide(result_0[1:], result_0[:-1])
        S_1 = np.divide(result_1[1:], result_1[:-1])
        S_2 = np.divide(result_2[1:], result_2[:-1])
        S_3 = np.divide(result_3[1:], result_3[:-1])
        se_out = np.vstack((S_0, S_1, S_2, S_3))
        se_out = se_out.T
        return se_out

    def cal_weight(self, y):
        y = y + np.ones([5, 4])
        y0 = np.prod(y[:, 0])
        y1 = np.prod(y[:, 1])
        y2 = np.prod(y[:, 2])
        y3 = np.prod(y[:, 3])
        if np.min([y0, y1, y2, y3]) == np.max([y0, y1, y2, y3]):
            y_weight = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            m = np.min([y0, y1, y2, y3])
            s = y0+y1+y2+y3
            Y0 = (y0 - m) / (s - 4 * m)
            Y1 = (y1 - m) / (s - 4 * m)
            Y2 = (y2 - m) / (s - 4 * m)
            Y3 = (y3 - m) / (s - 4 * m)
            y_weight = np.array([Y0, Y1, Y2, Y3])

        return y_weight


    def generate_data(self, x, flag):
        refx = x[0:-1, :]
        # refy = self.cal_weight(x[-1, :]) # 计算权重向量
        refy = self.cal_weight(x[-5:, :])

        mamldata.x_seeds.append(refx.T.tolist())
        mamldata.y_seeds.append(refy.T.tolist())
        return 0

    def generate_rp(self, x, flag):
        refrp = x[-1, :]
        mamldata.rp_seeds.append(refrp)

    # def vmd_decompotion(self, x, flag):
    #     # flag: 1判断是对训练集数据进行分解；0对测试集数据进行分解
    #     # 对x的每一列进行vmd分解最终揉成一个矩阵
    #     eemd = EEMD()
    #     IMFs = np.zeros([x.shape[0], 4*(self.nIMFs+1)]) #4为策略数量,由于还有一个残差，所以需要+1
    #     for i in range(x.shape[1]):
    #         u = eemd.eemd(x[:, i], range(x.shape[0]), self.nIMFs)
    #         u = u.T
    #         IMFs[:, (self.nIMFs+1)*i:(self.nIMFs+1)*(i+1)] = u
    #     if flag == 1:
    #         IMFs = IMFs
    #         self.roll_np(IMFs, self.generate_data, self.p + 1)
    #     else:
    #         self.generate_data(IMFs[-(self.p + 1):, :], flag)
    #     print("#####################")
    #     return 0

    def roll_np(self, df: pd.DataFrame, apply_func: callable, window: int, **kwargs):

        # move index to values
        v = df

        dim0, dim1 = v.shape
        stride0, stride1 = v.strides

        stride_values = stride(v, (dim0 - (window - 1), window, dim1), (stride0, stride0, stride1))

        # result_values = np.full((dim0, return_col_num), np.nan)

        for idx, values in enumerate(stride_values, window - 1):
            # values : col 1 is index, other is value
            # result_values[idx,] =
            if idx == window-1:
                flag = 1
            else:
                flag = 0
            apply_func(values, flag, **kwargs)
        return 0

    def roll_np_pw(self, df: np.array, window: int, **kwargs):

        # move index to values
        v = df

        dim0, dim1, dim2 = v.shape
        stride0, stride1, stride2 = v.strides

        stride_values = stride(v, (dim1 - (window - 1), dim0, window, dim2), (stride1, stride0, stride1,stride2))

        # result_values = np.full((dim0, return_col_num), np.nan)

        for idx, values in enumerate(stride_values, window - 1):
            refpw = values[:, -1, :]
            mamldata.pw_seeds.append(refpw)

        return 0

    def generate_traintest(self):
        se_data = self.readdata()
        # se_data.to_csv(ARmrnn.output_folder + 'stocksprice.returns.csv')
        no_port = se_data.shape[1]
        se_data_len = se_data.shape[0]
        divided_point = int(se_data_len * self.divide)  # The point is dividing the train and test data



        mamldata.x_seeds = []
        mamldata.y_seeds = []
        mamldata.pw_seeds = []
        mamldata.rp_seeds = []
        # self.roll_np(se_data, self.vmd_decompotion, se_data_len-divided_point)
        self.roll_np(se_data, self.generate_data, self.p + 1)
        self.roll_np_pw(self.port_pw, self.p + 1)
        self.roll_np(self.port_rp, self.generate_rp, self.p + 1)
        size = len(mamldata.x_seeds[0][0])

        x_train = (mamldata.x_seeds[:-divided_point])
        x_test = (mamldata.x_seeds[-divided_point:])

        y_train = (mamldata.y_seeds[:-divided_point])
        y_test = (mamldata.y_seeds[-divided_point:])

        pw_train = (mamldata.pw_seeds[:-divided_point])
        pw_test = (mamldata.pw_seeds[-divided_point:])

        rp_train = (mamldata.rp_seeds[:-divided_point])
        rp_test = (mamldata.rp_seeds[-divided_point:])

        # x_train = np.array(x_train).reshape(se_data_len - self.p - divided_point, no_port*(self.nIMFs+1), size)
        x_train = np.array(x_train).reshape(se_data_len - self.p - divided_point, no_port, size)
        y_train = np.array(y_train).reshape(se_data_len - self.p - divided_point, no_port, 1)
        pw_train = np.array(pw_train).reshape(se_data_len - self.p - divided_point, no_port, self.nstock)
        rp_train = np.array(rp_train).reshape(se_data_len - self.p - divided_point, self.nstock, 1)
        
        # x_test = np.array(x_test).reshape(divided_point, no_port*(self.nIMFs+1), size)
        x_test = np.array(x_test).reshape(divided_point, no_port, size)
        y_test = np.array(y_test).reshape(divided_point, no_port, 1)
        pw_test = np.array(pw_test).reshape(divided_point, no_port, self.nstock)
        rp_test = np.array(rp_test).reshape(divided_point, self.nstock, 1)



        return x_train, y_train, pw_train, rp_train, x_test, y_test, pw_test, rp_test

    def get_one_task_data(self):
        x_train, y_train, pw_train, rp_train, x_test, y_test, pw_test, rp_test = self.data
        if self.train == 1:
            x_data = x_train
            y_data = y_train
            pw_data = pw_train
            rp_data = rp_train
        else:
            x_data = x_test
            y_data = y_test
            pw_data = pw_test
            rp_data = rp_test
        n_task = x_data.shape[0] - args.k_shot - args.q_query
        random_index = random.randint(0, n_task)

        support_x = []
        support_pw = []
        support_rp = []
        support_y = []
        query_x = []
        query_y = []
        query_pw = []
        query_rp = []

        for i in range(random_index, random_index + args.k_shot):
            support_x.append(x_data[i])
            support_y.append(y_data[i])
            support_pw.append(pw_data[i])
            support_rp.append(rp_data[i])

        for j in range(random_index + args.k_shot, random_index + args.k_shot + args.q_query):
            query_x.append(x_data[j])
            query_y.append(y_data[j])
            query_pw.append(pw_data[j])
            query_rp.append(rp_data[j])

        return np.array(support_x), np.array(support_y), np.array(support_pw), np.array(support_rp), np.array(query_x), np.array(query_y), np.array(query_pw), np.array(query_rp)

    def get_one_batch(self):
        """
        获取一个batch的样本，这里一个batch中是以task为个体
        :return: k_shot_data, q_query_data
        """
        while True:
            batch_support_x = []
            batch_support_y = []
            batch_support_pw = []
            batch_support_rp = []
            batch_query_x = []
            batch_query_y = []
            batch_query_pw = []
            batch_query_rp = []

            for i in range(self.meta_batch_size):
                support_x, support_y, support_pw, support_rp, query_x, query_y, query_pw, query_rp = self.get_one_task_data()
                batch_support_x.append(support_x)
                batch_support_y.append(support_y)
                batch_support_pw.append(support_pw)
                batch_support_rp.append(support_rp)
                batch_query_x.append(query_x)
                batch_query_y.append(query_y)
                batch_query_pw.append(query_pw)
                batch_query_rp.append(query_rp)

            yield np.array(batch_support_x).reshape(self.meta_batch_size, args.k_shot,
                                                    self.nport, self.p), np.array(
                batch_support_y).reshape(self.meta_batch_size, args.k_shot, self.nport, 1), np.array(
                batch_support_pw).reshape(self.meta_batch_size, args.k_shot, self.nport, self.nstock), \
                  np.array(batch_support_rp).reshape(self.meta_batch_size, args.k_shot, self.nstock, 1), \
                  np.array(batch_query_x).reshape(self.meta_batch_size, args.q_query,
                                                  self.nport, self.p), np.array(
                batch_query_y).reshape(self.meta_batch_size, args.q_query, self.nport, 1),np.array(
                batch_query_pw).reshape(self.meta_batch_size, args.q_query, self.nport, self.nstock), \
                  np.array(batch_query_rp).reshape(self.meta_batch_size, args.q_query, self.nstock, 1),
