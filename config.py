# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--summary_path', type=str,
                    default="./summary",
                    help='The directory of the summary writer.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of task per train batch.')
parser.add_argument('--val_batch_size', type=int, default=16,
                    help='Number of task per test batch.')
parser.add_argument('--epochs', type=int, default=10,
                    help='The training epochs.')
parser.add_argument('--inner_lr', type=float, default=0.0003,
                    help='The learning rate of of the support set.')
parser.add_argument('--outer_lr', type=float, default=0.0003,#0.001,
                    help='The learning rate of of the query set.')

parser.add_argument('--k_shot', type=int, default=20,#250
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=1,
                    help='The number of query set image for every task.')
parser.add_argument('--nIMFs', type=int, default=4,
                    help='The nmber of nIMFs')

args = parser.parse_args()
