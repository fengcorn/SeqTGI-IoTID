#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import time
import threading
import psutil
import os

class ProductQuantization:
    def __init__(self, n_clusters=8):
        """
        参数:
            n_clusters: 子向量的数量（默认为8）。
        """
        self.n_clusters = n_clusters
        self.subvector_size = None
        self.codebooks = []
        self.quantized_data = None

    def fit(self, data):
        """
        对数据进行训练，构建多个子向量的聚类代码本。
        参数:
            data: 输入数据，形状为 (num_samples, feature_dim)。
        """
        # 确定每个子向量的大小
        self.subvector_size = data.shape[1] // self.n_clusters
        self.codebooks = []

        # 对每个子向量进行 KMeans 聚类
        for i in range(self.n_clusters):
            subvectors = data[:, i * self.subvector_size:(i + 1) * self.subvector_size]
            kmeans = KMeans(n_clusters=256, random_state=42)
            kmeans.fit(subvectors)
            self.codebooks.append(kmeans)

        # 对数据进行量化
        self.quantized_data = self.quantize(data)

    def quantize(self, data):
        """
        对数据进行量化，将每个子向量映射到最近的聚类中心。
        参数:
            data: 输入数据，形状为 (num_samples, feature_dim)。
        返回:
            量化后的数据。
        """
        quantized_data = []
        for i in range(self.n_clusters):
            subvectors = data[:, i * self.subvector_size:(i + 1) * self.subvector_size]
            quantized_subvectors = self.codebooks[i].predict(subvectors)
            quantized_data.append(quantized_subvectors)

        return np.column_stack(quantized_data)

    def search(self, query):
        """
        在量化后的数据中查找与查询向量最近的邻居。
        参数:
            query: 查询向量，形状为 (1, feature_dim)。
        返回:
            最近邻的索引。
        """
        quantized_query = self.quantize(query.reshape(1, -1))
        distances = np.sum((self.quantized_data - quantized_query) ** 2, axis=1)
        nearest_neighbor_index = np.argmin(distances)
        return nearest_neighbor_index

def get_cpu_mem():
    """
    获取当前进程的 CPU 和内存使用情况。
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu_percent = p.cpu_percent(interval=0.1) / psutil.cpu_count()
    mem_percent = p.memory_percent()
    print("pid:", pid)
    print("cpu:{:.8f}%, mem:{:.4f}%".format(cpu_percent, mem_percent))

def monitor():
    """
    监控系统的 CPU 和内存使用情况。
    """
    time.sleep(5)
    for _ in range(50):
        get_cpu_mem()
        time.sleep(0.5)
    print('监控线程结束')

if __name__ == '__main__':
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # 测试构建索引和搜索性能
    time.sleep(5)
    for n in [100]:
        m = 45000  # 数据库指纹数量
        data = 2 * np.random.random([m, 64]).astype(np.float32) - 1
        query = 2 * np.random.random([n, 64]).astype(np.float32) - 1

        t1 = time.time()
        pq = ProductQuantization(n_clusters=8)
        pq.fit(data)
        t2 = time.time()

        print("查询向量:")
        print(query)
        for i in range(n):
            nearest_neighbor_index = pq.search(query[i])
            nearest_neighbor_vector = data[nearest_neighbor_index]
        t3 = time.time()

        print("数据库指纹数：", m)
        print("查找指纹数：", n)
        print("PQ构建时间: {:.2f} 秒".format(t2 - t1))
        print("PQ搜索时间: {:.2f} 秒".format(t3 - t2))

    # 数据集评估
    preds = np.array([])
    data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_predict_result.npy')
    raw_data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_predict_result_raw.npy')
    y_raw = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_y_raw.npy')
    y_train = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex2_dscgru_y_train.npy')

    pq = ProductQuantization()
    pq.fit(data)

    for i in range(len(y_raw)):
        nearest_neighbor_index = pq.search(raw_data[i])
        preds = np.append(preds, y_train[nearest_neighbor_index])

    print(preds)
    print(classification_report(y_raw, preds, digits=3))