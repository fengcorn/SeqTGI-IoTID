#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Union
from sklearn.metrics import classification_report
import time
import threading
import psutil
import os

class EuclideanLSH:
    def __init__(self, tables_num: int, bucket_width: float, depth: int):
        """
        初始化LSH类

        :param tables_num: 哈希表的个数
        :param bucket_width: 桶宽，a越大，被纳入同个位置的向量就越多
        :param depth: 向量的维度
        """
        self.tables_num = tables_num
        self.bucket_width = bucket_width
        self.depth = depth
        # 随机生成哈希表的投影向量和偏移量
        self.R = np.random.random([depth, tables_num])
        self.b = np.random.uniform(0, bucket_width, [1, tables_num])
        # 初始化空的哈希表
        self.hash_tables = [dict() for _ in range(tables_num)]

    def _hash(self, inputs: Union[List[List], np.ndarray]) -> np.ndarray:
        """
        将向量映射到对应的哈希表索引

        :param inputs: 输入的单个或多个向量
        :return: 哈希值矩阵，每一行代表一个向量的所有哈希值
        """
        inputs = np.array(inputs)
        # H(V) = |V·R + b| / a
        hash_val = np.floor(np.abs(np.matmul(inputs, self.R) + self.b) / self.bucket_width)
        return hash_val

    def insert(self, inputs: Union[List[List], np.ndarray]):
        """
        将向量插入到所有哈希表中

        :param inputs: 要插入的向量或向量列表
        """
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        hash_index = self._hash(inputs)
        for vector, indices in zip(inputs, hash_index):
            for i, key in enumerate(indices):
                self.hash_tables[i].setdefault(key, []).append(tuple(vector))

    def query(self, inputs: Union[List, np.ndarray], top_k: int = 20) -> List[tuple]:
        """
        查询与输入向量相似的向量，返回相似度最高的top_k个

        :param inputs: 查询向量
        :param top_k: 返回的最相似向量个数
        :return: 最相似的向量列表
        """
        inputs = np.array(inputs)
        hash_val = self._hash(inputs).ravel()
        candidates = set()

        # 收集所有哈希表中对应索引的向量
        for i, key in enumerate(hash_val):
            candidates.update(self.hash_tables[i].get(key, []))

        # 根据欧式距离排序
        candidates = sorted(candidates, key=lambda x: self.euclidean_distance(x, inputs))
        return candidates[:top_k]

    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        计算欧式距离

        :param x: 向量x
        :param y: 向量y
        :return: 欧式距离
        """
        return np.sqrt(np.sum(np.power(x - y, 2)))

def get_cpu_mem():
    """
    获取当前进程的CPU和内存使用率
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu_percent = p.cpu_percent(interval=0.1) / psutil.cpu_count()
    mem_percent = p.memory_percent()
    print(f"pid: {pid}, cpu: {cpu_percent:.8f}%, mem: {mem_percent:.4f}%")

def monitor():
    """
    监控系统资源使用情况
    """
    time.sleep(5)
    for _ in range(50):
        get_cpu_mem()
        time.sleep(0.5)
    print('监控线程结束')

if __name__ == '__main__':
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    # 构建索引并测试搜索性能
    m = 50000  # 数据库指纹数
    n = 100    # 查找指纹数
    data = 2 * np.random.random([m, 64]).astype(np.float32) - 1
    query = 2 * np.random.random([n, 64]).astype(np.float32) - 1

    t1 = time.time()
    lsh = EuclideanLSH(10, 1, 64)
    lsh.insert(data)
    t2 = time.time()

    for i in range(n):
        lsh.query(query[i], 7)
    t3 = time.time()

    print("数据库指纹数：", m)
    print("查找指纹数：", n)
    print("LSH构建时间: {:.2f}s".format(t2 - t1))
    print("LSH搜索时间: {:.2f}s".format(t3 - t2))

    # 数据集评价
    preds = []
    data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_predict_result.npy')
    raw_data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_predict_result_raw.npy')
    y_raw = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_y_raw.npy')
    y_train = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_y_train.npy')

    lsh = EuclideanLSH(10, 1, 64)
    lsh.insert(data)

    for raw_vector in raw_data:
        res = lsh.query(raw_vector, 7)
        if res:
            closest_vector = res[0]
            # 找到最接近向量的索引
            match_index = np.where((data == closest_vector).all(axis=1))[0][0]
            preds.append(y_train[match_index])

    print(classification_report(y_raw, preds, digits=3))