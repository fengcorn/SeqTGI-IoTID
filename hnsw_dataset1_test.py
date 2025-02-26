#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nmslib
import datetime
import time
import threading
import psutil
import os
from functools import wraps
from sklearn.metrics import classification_report, confusion_matrix

def func_execute_time(func):
    """装饰器：计算函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).microseconds // 1000
        print(f"Function '{func.__name__}' executed in {duration:.4f} ms")
        return result
    return wrapper

def create_indexer(data, index_threads, m, ef):
    """基于数据向量构建索引"""
    index = nmslib.init(method="hnsw", space="l2")
    index.addDataPointBatch(data)
    index_params = {
        "indexThreadQty": index_threads,
        "M": m,
        "efConstruction": ef,
        "post": 2
    }
    index.createIndex(index_params, print_progress=True)
    index_path = f"data_{index_threads}_{m}_{ef}.hnsw"
    index.saveIndex(index_path)
    print(f"Index saved to {index_path}")

def load_indexer(index_path, ef_search=300):
    """加载构建好的向量索引文件"""
    indexer = nmslib.init(method="hnsw", space="l2")
    indexer.loadIndex(index_path)
    indexer.setQueryTimeParams({"efSearch": ef_search})
    return indexer

@func_execute_time
def search_vec_top_n(indexer, vectors, top_n=7):
    """使用索引文件完成向量查询"""
    ids, distances = indexer.knnQuery(vectors, k=top_n)
    return ids, distances

def monitor_system_resources():
    """监控系统资源使用情况"""
    pid = os.getpid()
    process = psutil.Process(pid)
    print(f"Monitoring process with PID: {pid}")
    for _ in range(50):
        cpu_percent = process.cpu_percent(interval=0.1) / psutil.cpu_count()
        mem_percent = process.memory_percent()
        print(f"CPU: {cpu_percent:.8f}%, Memory: {mem_percent:.4f}%")
        time.sleep(0.5)
    print("Monitoring thread finished.")

def evaluate_dataset(indexer, raw_data, y_raw, y_train):
    """评估数据集上的搜索性能"""
    preds = []
    for vector in raw_data:
        ids, _ = search_vec_top_n(indexer, vector)
        preds.append(y_train[ids[0]])
    preds = np.array(preds)
    print("Classification Report:")
    print(classification_report(y_raw, preds, digits=3))

    #confusion_mat = confusion_matrix(y_raw, preds, normalize='true')
    #np.save('/home/fengcorn/code/SeqTGI-IoTID/confusion_sentinel.npy', confusion_mat)
    #print(search_vec_top_n(indexer, data[:10]))
    #print(search_vec_top_n(indexer, data[-10:]))

def main():
    # 启动资源监控线程
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()
    time.sleep(5)

    # 构建索引并测试搜索性能
    m, n = 50000, 100  # 数据库大小和查询数量
    data = 2 * np.random.random([m, 64]).astype(np.float32) - 1
    query = 2 * np.random.random([n, 64]).astype(np.float32) - 1

    print("Building index...")
    t1 = time.time()
    create_indexer(data, index_threads=10, m=5, ef=300)
    indexer = load_indexer("data_10_5_300.hnsw")
    t2 = time.time()

    print("Searching...")
    for i in range(n):
        search_vec_top_n(indexer, query[i])
    t3 = time.time()

    print(f"\nDatabase size: {m}")
    print(f"Query size: {n}")
    print(f"HNSW Index Build Time: {t2 - t1:.2f} seconds")
    print(f"HNSW Search Time: {t3 - t2:.2f} seconds")

    # 数据集评估
    print("\nEvaluating dataset...")
    raw_data_path = '/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_predict_result_raw.npy'
    y_raw_path = '/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_y_raw.npy'
    y_train_path = '/home/fengcorn/code/SeqTGI-IoTID/Ex1_dscgru_y_train.npy'

    raw_data = np.load(raw_data_path)
    y_raw = np.load(y_raw_path)
    y_train = np.load(y_train_path)

    evaluate_dataset(indexer, raw_data, y_raw, y_train)

if __name__ == '__main__':
    main()