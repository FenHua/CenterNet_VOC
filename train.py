#!/usr/bin/env python
import os
import json
import torch
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback
import numpy as np
from tqdm import tqdm
from utils import stdout_to_tqdm
from db.datasets import datasets
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool

# 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
# 来达到优化运行效率的问题，backends 后端的
torch.backends.cudnn.enabled   = True  # 开启cudnn
torch.backends.cudnn.benchmark = True


# 输入函数
def parse_args():
    parser = argparse.ArgumentParser(description="Train CenterNet")         # 输入器
    parser.add_argument("cfg_file", help="config file", type=str)           # 配置文件
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)                                # 迭代的起始位置
    parser.add_argument("--threads", dest="threads", default=4, type=int)   # 线程数量
    args, unparsed = parser.parse_known_args()  # 在接受到多余的命令行参数时不报错，返回一个保存着余下的命令行字符的list
    return args


# 取数据
def prefetch_data(db, queue, sample_data, data_aug):
    # 取数据
    ind = 0
    print("start prefetching data...")  # prefetch 预读; 数据预取;
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e


# 从内存中获取数据
def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return


# 分配工作到各个线程
def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

# 训练函数
def train(training_dbs, validation_db, start_iter=0):
    learning_rate    = system_configs.learning_rate   # 学习率
    max_iteration    = system_configs.max_iter        # 最大迭代次数
    pretrained_model = system_configs.pretrain        # 预训练模型
    snapshot         = system_configs.snapshot        # 每隔snapshot进行参数保存
    val_iter         = system_configs.val_iter        # 每隔val_iter进行验证操作
    display          = system_configs.display         # 每隔display次进行loss输出
    decay_rate       = system_configs.decay_rate      # 学习率衰减
    stepsize         = system_configs.stepsize        # 学习率衰减的起始步
    # 获取每个数据集的大小
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)
    # 队列存储数据用于训练
    training_queue   = Queue(system_configs.prefetch_size)
    validation_queue = Queue(5)
    # 队列存储固定数据用于训练
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)
    # 加载数据采样函数
    data_file   = "sample.{}".format(training_dbs[0].data)
    sample_data = importlib.import_module(data_file).sample_data
    # 分配资源用于平行读取
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data, True)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)   # 验证数据集的并行
    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()
    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()
    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()
    # 创建模型
    print("building model...")
    nnet = NetworkFactory(training_dbs[0])

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))   # 根据当前的起始点来计算学习率

        nnet.load_params(start_iter)  # 加载起始点处的参数
        nnet.set_lr(learning_rate)    # 设置学习率
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)    # 设置学习率
    # 开始训练
    print("training start...")
    nnet.cuda()        # GPU
    nnet.train_mode()  # 训练模式
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)    # 训练数据
            training_loss, focal_loss, pull_loss, push_loss, regr_loss = nnet.train(**training)   # 获取loss
            if display and iteration % display == 0:
                print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
                print("focal loss at iteration {}:    {}".format(iteration, focal_loss.item()))
                print("pull loss at iteration {}:     {}".format(iteration, pull_loss.item())) 
                print("push loss at iteration {}:     {}".format(iteration, push_loss.item()))
                print("regr loss at iteration {}:     {}".format(iteration, regr_loss.item()))
                #print("cls loss at iteration {}:      {}\n".format(iteration, cls_loss.item()))
            del training_loss, focal_loss, pull_loss, push_loss, regr_loss     # 删除操作
            # 验证操作
            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()     # 验证模型
                validation = pinned_validation_queue.get(block=True)   # 验证集
                validation_loss = nnet.validate(**validation)          # 验证误差
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                nnet.train_mode()    # 切换到训练模式
            if iteration % snapshot == 0:
                nnet.save_params(iteration)   # 保存参数
            if iteration % stepsize == 0:
                learning_rate /= decay_rate   # 学习率进行衰减
                nnet.set_lr(learning_rate)

    # 训练结束后发送信号杀死线程
    training_pin_semaphore.release()
    validation_pin_semaphore.release()
    # 终止数据预读进程
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()  # 输入解析器
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")  # CenterNet配置文件，模型深度，多类别
    with open(cfg_file, "r") as f:
        configs = json.load(f)    # 加载配置文件
            
    configs["system"]["snapshot_name"] = args.cfg_file    # 当前模型的名称，如 CenterNet-52,CenterNet-104
    system_configs.update_config(configs["system"])       # 更新系统配置文件
    train_split = system_configs.train_split              # 训练数据集
    val_split   = system_configs.val_split                # 验证数据集
    # 加载所有数据集
    print("loading all datasets...")
    dataset = system_configs.dataset
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    validation_db = datasets[dataset](configs["db"], val_split)
    print("system config...")
    pprint.pprint(system_configs.full)       # 输出系统配置
    print("db config...")
    pprint.pprint(training_dbs[0].configs)   # 输出训练数据集配置
    print("len of db: {}".format(len(training_dbs[0].db_inds)))   # 输出训练数据集的长度
    train(training_dbs, validation_db, args.start_iter)
