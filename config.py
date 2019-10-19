import os
import numpy as np

# 网络训练或者测试时的一系列参数
class Config:
    def __init__(self):
        self._configs = {}
        self._configs["dataset"]           = None
        self._configs["sampling_function"] = "kp_detection"

        # 训练时的一系列配置
        self._configs["display"]           = 20       # 每隔display次进行一次loss输出
        self._configs["snapshot"]          = 500      # 每隔snapshot进行参数的保存
        self._configs["stepsize"]          = 2000     # 学习率进行调整的起始点
        self._configs["learning_rate"]     = 0.00025  # 学习率
        self._configs["decay_rate"]        = 10       # 学习率衰减
        self._configs["max_iter"]          = 3000     # 最大迭代次数
        self._configs["val_iter"]          = 100      # 每隔val_iter次进行一次验证操作
        self._configs["batch_size"]        = 1        # batch size大小
        self._configs["snapshot_name"]     = None     # 模型保存的名称
        self._configs["prefetch_size"]     = 100      # 预读取数据的量
        self._configs["weight_decay"]      = False    # 参数衰减
        self._configs["weight_decay_rate"] = 1e-5     # 参数衰减率
        self._configs["weight_decay_type"] = "l2"     # 参数衰减类型，默认是l2
        self._configs["pretrain"]          = None     # 预训练
        self._configs["opt_algo"]          = "adam"   # 优化器 adam
        self._configs["chunk_sizes"]       = None

        # 各种数据目录
        self._configs["data_dir"]   = "data"          # 数据集
        self._configs["cache_dir"]  = "cache"         # cache目录
        self._configs["config_dir"] = "config"        # config目录，存储各种类别的网络模型
        self._configs["result_dir"] = "results"       # 结果存放目录

        # 数据集划分
        self._configs["train_split"] = "trainval"     # 训练数据集
        self._configs["val_split"]   = "minival"      # 验证数据集
        self._configs["test_split"]  = "testdev"      # 测试数据集

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)


    @property
    def chunk_sizes(self):
        return self._configs["chunk_sizes"]

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def weight_decay_type(self):
        return self._configs["weight_decay_type"]

    @property
    def prefetch_size(self):
        return self._configs["prefetch_size"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def weight_decay_rate(self):
        return self._configs["weight_decay_rate"]

    @property
    def weight_decay(self):
        return self._configs["weight_decay"]

    @property
    def result_dir(self):
        result_dir = os.path.join(self._configs["result_dir"], self.snapshot_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + "_{}.pkl")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def decay_rate(self):
        return self._configs["decay_rate"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def snapshot(self):
        return self._configs["snapshot"]

    @property
    def display(self):
        return self._configs["display"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

system_configs = Config()
