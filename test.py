#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import matplotlib
import numpy as np
matplotlib.use("Agg")                         # 不显示图片
from db.datasets import datasets              # 数据库
from config import system_configs             # 各种设置配置文件
from nnet.py_factory import NetworkFactory    # 模型
torch.backends.cudnn.benchmark = False        # cuda运行不是动态调整


# 输入操作
def parse_args():
    parser = argparse.ArgumentParser(description="Test CenterNet")             # 输入器
    parser.add_argument("cfg_file", help="config file", type=str)              # 网络结构配置文件
    parser.add_argument("--testiter", dest="testiter",help="test at iteration i",
                        default=None, type=int)                                # 采用第几次迭代后保存的模型进行测试
    parser.add_argument("--split", dest="split",help="which split to use",
                        default="validation", type=str)                        # 采用拿个文件夹的数据进行测试
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)     # suffix 后缀，用来给出更加具体的存储目录
    parser.add_argument("--debug", action="store_true")                        # 选择存储
    args = parser.parse_args()                                                 # 输入解析
    return args


# 创建文件夹
def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


# 检测函数
def test(db, split, testiter, debug=False, suffix=None):
    result_dir = system_configs.result_dir                                  # 结果存放目录
    result_dir = os.path.join(result_dir, str(testiter), split)             # 具体的结果存放子目录
    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)                       # 如果给了更具体的后缀描述，则加入
    make_dirs([result_dir])                                                 # 创建文件夹
    test_iter = system_configs.max_iter if testiter is None else testiter   # 如果没指定率第test_iter个模型，则默认选择最终模型
    print("loading parameters at iteration: {}".format(test_iter))          # 显示加载第test_iter次的模型参数
    print("building neural network...")                                     # 创建神经网络
    nnet = NetworkFactory(db)                                               # 构建网络，db是网络的一系列构造参数
    print("loading parameters...")
    nnet.load_params(test_iter)                                             # 加载已保存的网络模型
    test_file = "test.{}".format(db.data)                                   # 网络参数
    testing = importlib.import_module(test_file).testing                    # 检测函数
    nnet.cuda()
    nnet.eval_mode()                                                        # 训练模式
    testing(db, nnet, result_dir, debug=debug)                              # 检测


# 主函数
if __name__ == "__main__":
    args = parse_args()                                                     # 解析输入
    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")     # 没有后缀，则直接使用配置文件
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))   # 给定了后缀
    print("cfg_file: {}".format(cfg_file))          # 输入采用的网络模型名称
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = args.cfg_file   # 更新模型名称
    system_configs.update_config(configs["system"])      # 更新
    train_split = system_configs.train_split             # 训练数据集
    val_split   = system_configs.val_split               # 验证数据集
    test_split  = system_configs.test_split              # 测试数据集
    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]                                           # 给定的测试文件夹
    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    testing_db = datasets[dataset](configs["db"], split)    # 获取测试数据集
    print("system config...")
    pprint.pprint(system_configs.full)                      # 输出配置
    print("db config...")
    pprint.pprint(testing_db.configs)                       # 输出测试数据集的配置情况
    # 检测
    test(testing_db, args.split, args.testiter, args.debug, args.suffix)
