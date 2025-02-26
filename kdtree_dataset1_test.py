#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
import threading
import psutil
import os


class kdTree:
    def __init__(self, parent_node=None):
        """
        节点初始化
        """
        self.nodedata = None  # 当前节点的数据值，二维数据
        self.split = None  # 分割平面的方向轴序号, 0代表沿着x轴分割，1代表沿着y轴分割
        self.range = None  # 分割临界值
        self.left = None  # 左子树节点
        self.right = None  # 右子树节点
        self.parent = parent_node  # 父节点
        self.leftdata = None  # 保留左边节点的所有数据
        self.rightdata = None  # 保留右边节点的所有数据
        self.isinvted = False  # 记录当前节点是否被访问过

    def print(self):
        """
        打印当前节点信息
        """
        print(self.nodedata, self.split, self.range)

    def get_split_axis(self, all_data):
        """
        根据方差决定分割轴
        """
        var_all_data = np.var(all_data, axis=0)
        return np.argmax(var_all_data)  # 返回方差最大的轴

    def get_range(self, split_axis, all_data):
        """
        获取对应分割轴上的中位数据值大小
        """
        split_all_data = all_data[:, split_axis]
        med_index = len(split_all_data) // 2
        return np.sort(split_all_data)[med_index]

    def get_node_left_right_data(self, all_data):
        """
        将数据划分到左子树，右子树以及得到当前节点
        """
        left_mask = all_data[:, self.split] < self.range
        right_mask = ~left_mask

        self.leftdata = all_data[left_mask]
        self.rightdata = all_data[right_mask]

        # 选择中位数作为当前节点
        median_index = len(self.leftdata)
        if median_index < len(all_data):
            self.nodedata = all_data[median_index]

    def create_next_node(self, all_data):
        """
        迭代创建节点，生成kd树
        """
        if all_data.shape[0] == 0:
            print("Create kd tree finished!")
            return None

        self.split = self.get_split_axis(all_data)
        self.range = self.get_range(self.split, all_data)
        self.get_node_left_right_data(all_data)

        if self.leftdata.shape[0] != 0:
            self.left = kdTree(self)
            self.left.create_next_node(self.leftdata)

        if self.rightdata.shape[0] != 0:
            self.right = kdTree(self)
            self.right.create_next_node(self.rightdata)

    def plot_kd_tree(self):
        """
        在图上画出来树形结构的递归迭代过程
        """
        if self.parent is None:
            plt.figure(dpi=300)
            plt.xlim([0.0, 10.0])
            plt.ylim([0.0, 10.0])

        color = np.random.random(3)
        if self.left is not None:
            plt.plot([self.nodedata[0], self.left.nodedata[0]], [self.nodedata[1], self.left.nodedata[1]], '-o', color=color)
            plt.arrow(self.nodedata[0], self.nodedata[1], (self.left.nodedata[0] - self.nodedata[0]) / 2.0,
                      (self.left.nodedata[1] - self.nodedata[1]) / 2.0, color=color, head_width=0.2)
            self.left.plot_kd_tree()

        if self.right is not None:
            plt.plot([self.nodedata[0], self.right.nodedata[0]], [self.nodedata[1], self.right.nodedata[1]], '-o', color=color)
            plt.arrow(self.nodedata[0], self.nodedata[1], (self.right.nodedata[0] - self.nodedata[0]) / 2.0,
                      (self.right.nodedata[1] - self.nodedata[1]) / 2.0, color=color, head_width=0.2)
            self.right.plot_kd_tree()

    def div_data_to_left_or_right(self, find_data):
        """
        根据传入的数据将其分给左节点(0)或右节点(1)
        """
        return 0 if find_data[self.split] < self.range else 1

    def get_search_path(self, ls_path, find_data):
        """
        二叉查找到叶节点上
        """
        now_node = ls_path[-1]
        if now_node is None:
            return ls_path

        now_split = now_node.div_data_to_left_or_right(find_data)
        next_node = now_node.left if now_split == 0 else now_node.right

        while next_node is not None:
            ls_path.append(next_node)
            next_split = next_node.div_data_to_left_or_right(find_data)
            next_node = next_node.left if next_split == 0 else next_node.right

        return ls_path

    def get_nearest_node(self, find_data, min_dist, min_data):
        """
        回溯查找目标点的最近邻距离
        """
        ls_path = [self]
        self.get_search_path(ls_path, find_data)

        now_node = ls_path.pop()
        now_node.isinvted = True
        min_data = now_node.nodedata
        min_dist = np.linalg.norm(find_data - min_data)

        while ls_path:
            back_node = ls_path.pop()
            if back_node.isinvted:
                continue

            back_node.isinvted = True
            back_dist = np.linalg.norm(find_data - back_node.nodedata)
            if back_dist < min_dist:
                min_data = back_node.nodedata
                min_dist = back_dist

            if np.abs(find_data[back_node.split] - back_node.range) < min_dist:
                ls_path.append(back_node)
                if back_node.left.isinvted:
                    if back_node.right is not None:
                        ls_path.append(back_node.right)
                else:
                    if back_node.left is not None:
                        ls_path.append(back_node.left)

                ls_path = back_node.get_search_path(ls_path, find_data)
                now_node = ls_path.pop()
                now_node.isinvted = True
                now_dist = np.linalg.norm(find_data - now_node.nodedata)
                if now_dist < min_dist:
                    min_data = now_node.nodedata
                    min_dist = now_dist

        return min_dist, min_data

    def get_nearest_dist_by_exhaustive(self, test_array, find_data):
        """
        穷举法得到目标点的最近邻距离
        """
        min_data = test_array[0]
        min_dist = np.linalg.norm(find_data - min_data)

        for now_data in test_array:
            now_dist = np.linalg.norm(find_data - now_data)
            if now_dist < min_dist:
                min_dist = now_dist
                min_data = now_data

        print(f"Min distance: {min_dist}, Min data: {min_data}")
        return min_dist


def get_cpu_mem():
    """
    获取当前进程的CPU和内存使用情况
    """
    pid = os.getpid()
    p = psutil.Process(pid)
    cpu_percent = p.cpu_percent(interval=0.1) / psutil.cpu_count()
    mem_percent = p.memory_percent()
    print(f"CPU: {cpu_percent:.8f}%, MEM: {mem_percent:.4f}%")


def monitor():
    """
    监控CPU和内存使用情况
    """
    time.sleep(5)
    for _ in range(50):
        get_cpu_mem()
        time.sleep(0.5)
    print('Monitor thread over')


if __name__ == '__main__':
    # 构建索引并测试搜索性能
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    time.sleep(5)

    min_dist = 0  # 临时变量，存储最短距离
    min_data = np.zeros((64,))  # 临时变量，存储取到最短距离时对应的数据点

    for n in [100]:
        m = 50000  # 数据库指纹数
        data = 2 * np.random.random([m, 64]).astype(np.float32) - 1
        query = 2 * np.random.random([n, 64]).astype(np.float32) - 1

        t1 = time.time()
        my_kd_tree = kdTree()
        my_kd_tree.create_next_node(data)
        t2 = time.time()

        for i in range(n):
            min_dist, min_data = my_kd_tree.get_nearest_node(query[i], min_dist, min_data)

        t3 = time.time()
        print(f"数据库指纹数: {m}")
        print(f"查找指纹数: {n}")
        print(f"KDTree构建: {t2 - t1}")
        print(f"KDTree搜索: {t3 - t2}")

    # 数据集评估
    preds = np.array([])
    data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_predict_result.npy')
    raw_data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_predict_result_raw.npy')
    y_raw = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_y_raw.npy')
    y_train = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_y_train.npy')

    my_kd_tree = kdTree()
    my_kd_tree.create_next_node(data)

    for i in range(len(y_raw)):
        find_dist, find_data = my_kd_tree.get_nearest_node(raw_data[i], min_dist, min_data)
        ids = np.argwhere(data == find_data[0])[0][0]
        preds = np.append(preds, y_train[ids])

    print(preds)
    print(classification_report(y_raw, preds, digits=3))