from keras.models import Sequential
from keras.layers import Dense, LSTM, LeakyReLU, Dropout
import numpy as np
import time
import pandas as pd
from tensorflow.keras import optimizers, utils
import matplotlib.pyplot as plt
import tensorflow as tf
from datareader import mamldata
from config import args
from keras import backend as K

from universal import tools
from universal.algos import RMR
from universal.algos import CORN
from universal.algos import Anticor
from universal.algos import WMAMR
from universal.algos import PAMR
from universal.algos import EG
from universal.algos import ONS
from universal.algos import OLMAR
from universal.algos import CRP
from universal.algos import BestSoFar
from universal.algos import UP
from universal.algos import BNN
from universal.algos import BAH


class rnnmaml_e2e():

    def __init__(self, weight, equity, re_price, p=20, batch_size=3, divide=1 / 4, train=1,  nport=4):
        self.p = p  # 窗口大小
        self.meta_batch_size = batch_size
        self.divide = divide  # 测试集和训练集之比？
        self.train = train
        self.pw = weight
        self.eq = equity
        self.rp = re_price
        self.nstock = re_price.shape[1]
        self.nIMFs = args.nIMFs # eemd分解后IMF+残差数量
        self.nport = nport # 策略库的数量
        self.meta_model = self.getmodel(args.nIMFs, nport, p) # 构造模型

    def getmodel(self, nIMFs, nport, size):
        # 创建模型
        model = Sequential()
        # imf要改
        model.add(LSTM(64, input_shape=(nport, size), return_sequences=True))  # 返回所有节点的输出
        model.add(LSTM(32, return_sequences=False))  # 返回最后一个节点的输出
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu'))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(nport, activation='softmax'))
        return model
        # 输入层
        # input_layer = Input(shape=(nport, size))
        #
        # # 双层LSTM
        # lstm1 = LSTM(64, return_sequences=True)(input_layer)
        # lstm2 = LSTM(32, return_sequences=True)(lstm1)
        #
        # # 注意力机制
        # attention = Attention()([lstm2, lstm1])# 合并输出
        # merged = Concatenate(axis=-1)([lstm2, attention])
        #
        # # Dropout层
        # dropout = Dropout(0.1)(attention)
        #
        # # 全连接层
        # dense1 = Dense(16, activation='relu')(dropout)
        # dense2 = Dense(nport, activation='softmax')(dense1)
        #
        # # 创建模型
        # model = Model(inputs=input_layer, outputs=dense2)
        # return model


    def myloss_cumreturn(self, y_true, y_pred, pw, rp):
        w1, w2, w3, w4 = np.tile(y_pred[:, 0], (self.nstock, 1)).T, np.tile(y_pred[:, 1],
                                                                             (self.nstock, 1)).T, np.tile(y_pred[:, 2],
                                                                                                        (self.nstock,
                                                                                                         1)).T, np.tile(
            y_pred[:, 3], (self.nstock, 1)).T
        b = w1 * pw[:, 0, :] + w2 * pw[:, 1, :] + w3 * pw[:, 2, :] + w4 * pw[:, 3, :]
        b = np.array(b)
        tlb = b * rp[:,:,0]
        row_sums = tlb.sum(axis=1)
        tlb = tlb /row_sums[:, np.newaxis]
        cost = np.insert(np.sum(np.abs(b[1:] - tlb[:-1]), axis=1), 0, 1)
        cost = tf.convert_to_tensor(cost, dtype=tf.float32)
        diff = np.insert(np.sum(np.abs(y_pred[1:] - y_pred[:-1]), axis=1), 0, 0)
        # 前一项判断准确率，第二项使权重更为分散, 第三项减少交易费用
        return K.mean(K.square(y_true - y_pred), axis=-1) + 0.0000003 * K.mean(K.square(y_pred), axis=-1) + 0.0001 * cost
        # return -K.mean((y_true*y_pred), axis=-1) + 0.0003*K.mean(K.square(y_pred), axis=-1) #+ 0.01 * cost
        # return tf.convert_to_tensor(np.linalg.norm(y_pred-y_true, axis=1), dtype=tf.float32) + 0.0000000000001*K.mean(K.square(y_pred), axis=-1) + 0.01 * 0.0001 * cost + 0.001 * diff

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        """
        MAML一个batch的训练过程，在原权重的基础上下降一步
        :param train_data: 训练数据，以task为一个单位
        :param inner_optimizer: support set对应的优化器
        :param inner_step: 内部更新几个step
        :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
        :return: batch query loss
        """
        batch_loss = []
        task_weights = []

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()

        meta_support_x, meta_support_y, meta_support_pw, meta_support_rp, meta_query_x, meta_query_y, meta_query_pw, meta_query_rp = next(train_data) # 将数据分为支持集和查询集
        for support_x, support_y, support_pw, support_rp in zip(meta_support_x, meta_support_y, meta_support_pw, meta_support_rp):

            # 每个task都需要载入最原始的weights进行更新
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_x, training=True)
                    loss = self.myloss_cumreturn(np.squeeze(support_y), logits, support_pw, support_rp)
                    #loss = losses.mean_squared_error(np.squeeze(support_y), logits)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_x, query_y, query_pw, query_rp) in enumerate(zip(meta_query_x, meta_query_y, meta_query_pw, meta_query_rp)):
                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i])
                logits = self.meta_model(query_x, training=True)
                loss = self.myloss_cumreturn(np.squeeze(query_y), logits, query_pw, query_rp)
                #loss = losses.mean_squared_error(np.squeeze(query_y), logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)
            mean_loss = tf.reduce_mean(batch_loss)

        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)  ##计算loss关于变量的梯度
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss

    def trainmodel(self):
        train_data = mamldata(self.p, self.pw, self.eq, self.rp, batch_size=self.meta_batch_size, divide=self.divide, nIMFs = self.nIMFs, train = self.train)

        inner_optimizer = optimizers.Adam(args.inner_lr)
        outer_optimizer = optimizers.Adam(args.outer_lr)

        for e in range(args.epochs):
            train_progbar = utils.Progbar(train_data.steps)
            # 用于显示进度条
            #val_progbar = utils.Progbar(val_data.steps)
            print('\nEpoch {}/{}'.format(e + 1, args.epochs))

            train_meta_loss = []
            val_meta_loss = []
            for i in range(train_data.steps):
                batch_train_loss = self.train_on_batch(train_data.get_one_batch(), inner_optimizer, inner_step=5,
                                                       outer_optimizer=outer_optimizer)

                train_meta_loss.append(batch_train_loss)
                train_progbar.update(i + 1, [('loss', np.mean(train_meta_loss))])
            self.meta_model.save_weights("mamle2e.h5")

        return train_meta_loss

    def get_new_one(self,x_test, y_test, pw_test, rp_test, index):
        # 将集合划分为支持集和查询集
        support_x = []
        support_y = []
        support_pw = []
        support_rp = []
        query_x = []
        query_y = []
        query_pw = []
        query_rp = []

        for i in range(index, index + args.k_shot):
            support_x.append(x_test[i])
            support_y.append(y_test[i])
            support_pw.append(pw_test[i])
            support_rp.append(rp_test[i])

        for j in range(index + args.k_shot, index + args.k_shot + 1):
            query_x.append(x_test[j])
            query_y.append(y_test[j])
            query_pw.append(pw_test[j])
            query_rp.append(rp_test[j])

        return np.array(support_x), np.array(support_y), np.array(support_pw),np.array(support_rp), np.array(query_x), np.array(query_y), np.array(query_pw), np.array(query_rp)

    def runmodel(self):
        train_meta_loss = self.trainmodel()
        x_train, y_train, pw_train, rp_train, x_test, y_test, pw_test, rp_test = mamldata(self.p, self.pw, self.eq, self.rp, batch_size=self.meta_batch_size, divide=self.divide,
                 nIMFs=self.nIMFs, train=self.train).data

        test_size = y_test.shape[0] - args.k_shot
        y_query = y_test[-test_size:]
        optimizer = optimizers.Adam(args.inner_lr)
        predict = np.zeros((test_size, self.nport))
        inner_step = 10
        train_time = 0
        for i in range(0, test_size):
            start_time = time.time()
            support_x, support_y, support_pw, support_rp, query_x, query_y, query_pw, query_rp = self.get_new_one(x_test, y_test, pw_test, rp_test, i)
            # 每个task都需要载入最原始的weights进行更新
            self.meta_model.load_weights("mamle2e.h5")
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_x, training=True)
                    loss=self.myloss_cumreturn(np.squeeze(support_y), logits, support_pw, support_rp)
                    #loss = losses.mean_squared_error(np.squeeze(support_y), logits)
                    loss = tf.reduce_mean(loss)

                    grads = tape.gradient(loss, self.meta_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

                # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
                task_weights=(self.meta_model.get_weights())

            with tf.GradientTape() as tape:
                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights)
                logits = self.meta_model(query_x, training=True)
                val_loss = self.myloss_cumreturn(np.squeeze(query_y), logits, query_pw, query_rp)
                #val_loss = losses.mean_squared_error(np.squeeze(query_y), logits)
                val_loss = tf.reduce_mean(val_loss)
            #train = np.array([x_test[i].tolist()])
            #test = y_test[i]
            finish_time = time.time()
            train_time = train_time+finish_time-start_time
            predict[i]=logits.numpy()
        train_time=train_time/test_size
        return self.meta_model, predict, y_query, train_time, val_loss


# 初始化 rnnmaml 类的对象
# selected_port = ["BAH", "WAMAR", "Anticor", "BNN"]
selected_port = ["EG", "ONS", "Anticor", "RMR"]
data = pd.read_csv('./4DATA_2DATA/data/nyse_o.csv')
# data = pd.read_csv('./4DATA_2DATA/data/nyse_o.csv')
result1 = tools.quickrun(EG(), data)
result2 = tools.quickrun(ONS(), data)
result3 = tools.quickrun(Anticor(), data)
result4 = tools.quickrun(RMR(), data)
# result1 = tools.quickrun(BAH(), data)
# result2 = tools.quickrun(WMAMR(), data)
# result3 = tools.quickrun(Anticor(), data)
# result4 = tools.quickrun(BNN(), data)
# print("begin")
# result1 = tools.quickrun(BNN(), data)
# result2 = tools.quickrun(RMR(), data)
# result3 = tools.quickrun(Anticor(), data)
# result4 = tools.quickrun(UP(), data)
print("policy run end")
# 价格变化向量
re_price = np.zeros([data.shape[0]-1, data.shape[1]])
for i in range(data.shape[0]-1):
    re_price[i, :] = np.divide(np.array(data)[i+1, :], np.array(data)[i, :])

# weight4 = np.array(result4.weights)[:,:-1]
# 处理一下权重向量，使其属于\delta_nstock^+
weight_1 = np.array(result1.weights)
weight_2 = np.array(result2.weights)
weight_3 = np.array(result3.weights)
weight_4 = np.array(result4.weights)
# 将各策略的投资组合权重向量做处理
for i in range(weight_1.shape[0]):
    if np.sum(weight_1[i, :]) == 0:
        weight_1[i, :] = weight_1[i - 1, :]
    if np.sum(weight_2[i, :]) == 0:
        weight_2[i, :] = weight_2[i - 1, :]
    if np.sum(weight_3[i, :]) == 0:
        weight_3[i, :] = weight_3[i - 1, :]
    if np.sum(weight_4[i, :]) == 0:
        weight_4[i, :] = weight_4[i - 1, :]
    weight_1[i, :] = weight_1[i, :] / np.sum(weight_1[i, :])
    weight_2[i, :] = weight_2[i, :] / np.sum(weight_2[i, :])
    weight_3[i, :] = weight_3[i, :] / np.sum(weight_3[i, :])
    weight_4[i, :] = weight_4[i, :] / np.sum(weight_4[i, :])
weight = np.array([weight_1, weight_2, weight_3, weight_4])
weight = weight[:,1:,:]

data_test = pd.read_csv('./4DATA_2DATA/DATA_culmulative_wealth/nyse_o_test_cumulative_wealth_nocost.csv')[selected_port]
port_equity = pd.read_csv('./4DATA_2DATA/DATA_culmulative_wealth/nyse_o_test_cumulative_wealth.csv')
equity = pd.DataFrame(np.array(data_test.pct_change().values)[1:,:])
rnnmaml_instance = rnnmaml_e2e(weight, equity, re_price, p=15, batch_size=5, divide=1 / 4, train=1,  nport=4)

# 训练模型并获取测试集上的结果
train_meta_loss = rnnmaml_instance.trainmodel()
meta_model, predict, y_query, train_time, val_loss = rnnmaml_instance.runmodel()

# a = predict
# predict = y_query[:,:,0]
n = predict.shape[0]

weight_1 = weight_1[-n:]
weight_2 = weight_2[-n:]
weight_3 = weight_3[-n:]
weight_4 = weight_4[-n:]
#计算本文策略的权重
a = predict
max_values = np.max(predict, axis=1)
predict = (predict == max_values[:, np.newaxis]).astype(int)
num_stock = weight_1.shape[1]
w1, w2, w3, w4 = np.tile(predict[:,0], (num_stock, 1)).T, np.tile(predict[:,1], (num_stock, 1)).T, np.tile(predict[:, 2], (num_stock, 1)).T, np.tile(predict[:, 3], (num_stock, 1)).T
weight = w1*weight_1 + w2*weight_2 + w3*weight_3 + w4*weight_4
weight = np.array(weight)

# 计算相对价格
# re_price =  np.zeros([predict.shape[0], num_stock])
# for i in range(n):
#     re_price[n-i-1, :] = np.divide(np.array(data)[-i-1, :], np.array(data)[-i-2, :])
# weight = weight_1
re_price =  np.divide(np.array(data)[-n:, :], np.array(data)[-n-1:-1, :])
# 做出累计收益曲线
S = np.ones([predict.shape[0]+1])


cost = 0.0005
for i in range(predict.shape[0]):
    if i==0:
        S[1] = S[0] * np.sum(weight[0] * re_price[i, :]) * (1-cost)
    else:
        S[i + 1] = S[i] * np.sum(weight[i] * re_price[i, :]) * (1 - cost * np.linalg.norm(
            weight[i] - weight[i - 1] * re_price[i - 1, :]/np.sum(weight[i - 1] * re_price[i - 1, :]), ord=1))


# 做累计收益曲线对比图
plt.figure(figsize=(10, 6))
plt.plot(S, linewidth=3, label='Learning mixture policies strategy')
for column in port_equity.columns:
    equity = np.array(port_equity[column][-n-1:])/np.array(port_equity[column])[-n-1]
    if column in selected_port:
        plt.plot(equity, linestyle='--', linewidth=2, alpha=0.7,label=column)
    # else:
    #     plt.plot(equity, label=column)
# for column in data_test.columns:
#     equity = np.array(data_test[column][-n-1:])/np.array(data_test[column])[-n-1]
#     plt.plot(equity, label=column)

plt.yscale('log')
plt.xlabel('Day')
plt.ylabel('Culmulative wealth')
plt.title('Comparison of Cumulative Wealth for Different Methods(nyse_o)')
plt.legend()
plt.show()
plt.savefig("./result/nyse_o_new3.jpg")

#存一下predict和S
# 转换为 DataFrame
pre = np.vstack((predict,np.zeros((1, predict.shape[1]))))
merged_array = np.hstack((pre, S.reshape(S.shape[0],1)))

np.savetxt('./result/nyse_o_new3.csv', merged_array, delimiter=',')


print(np.argmax(predict,axis=1))
print(np.argmax(y_query[:,:,0],axis=1))