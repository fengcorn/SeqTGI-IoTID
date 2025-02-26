#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/11
# @Author  : Feng Yuming

import numpy as np
import nmslib
import datetime
from functools import wraps
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
 
 
def func_execute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        duration_time = (end_time - start_time).microseconds // 1000
        # print("execute function %s, elapse time %.4f ms" % (func.__name__, duration_time))
        return res
    return wrapper
 
 
def create_indexer(vec, index_thread, m, ef):
    """
    基于数据向量构建索引
    :param vec: 原始数据向量
    :param index_thread: 线程数
    :param m: 
    :param ef: 
    :return:
    """
    index = nmslib.init(method="hnsw", space="l2")
    #index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(vec)
    INDEX_TIME_PARAMS = {
        "indexThreadQty": index_thread,
        "M": m,
        "efConstruction": ef,
        "post": 2
    }
    index.createIndex(INDEX_TIME_PARAMS, print_progress=True)
    index.saveIndex("data_%d_%d_%d.hnsw" % (index_thread, m, ef))
 
 
def load_indexer(index_path, ef_search=300):
    """
    加载构建好的向量索引文件
    :param index_path: 索引文件地址
    :param ef_search: 查询结果参数
    :return:
    """
    indexer = nmslib.init(method="hnsw", space="l2")
    # indexer = nmslib.init(method="hnsw", space="cosinesimil")
    indexer.loadIndex(index_path)
    indexer.setQueryTimeParams({"efSearch": ef_search})
    return indexer
 
 
@func_execute_time
def search_vec_top_n(indexer, vecs, top_n=7, threads=4):
    """
    使用构建好的索引文件完成向量查询
    :param indexer: 索引
    :param vecs: 待查询向量
    :param top_n: 返回前top_n个查询结果
    :param threads:
    :return:
    """
    # neighbours = indexer.knnQueryBatch(vecs, k=top_n, num_threads=threads)
    # return neighbours
    ids, distances = indexer.knnQuery(vecs, k=top_n)
    return ids, distances
 
def create_hnsw_indexer():
    data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_test_allrealdevice_predict_result.npy')
    print(data.shape)
    create_indexer(data, 10, 5, 300)
 
if __name__ == '__main__':

    create_hnsw_indexer()
    indexer = load_indexer("data_10_5_300.hnsw")

    raw_data = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_test_allrealdevice_predict_result_raw.npy')
    y_raw = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_test_allrealdevice_y_raw.npy')
    y_train = np.load('/home/fengcorn/code/SeqTGI-IoTID/Ex1_test_allrealdevice_y_train.npy')
    preds = np.array([])
    for i in range(len(y_raw)):
        ids, distances = search_vec_top_n(indexer, raw_data[i])
        preds = np.append(preds, y_train[ids[0]])
    print(classification_report(y_raw, preds, digits=3))

    # # Reorder y_raw
    y_raw_reordered = np.concatenate((y_raw[y_raw != 8], y_raw[y_raw == 8]))
    y_raw_reordered = np.concatenate((y_raw_reordered[y_raw_reordered != 0], y_raw_reordered[y_raw_reordered == 0]))

    # Reorder preds
    preds_reordered = np.concatenate((preds[preds != 8], preds[preds == 8]))
    preds_reordered = np.concatenate((preds_reordered[preds_reordered != 0], preds_reordered[preds_reordered == 0]))

    confusion_mat = confusion_matrix(y_raw_reordered, preds_reordered, normalize='true')
    print(confusion_mat)
    np.save('/home/fengcorn/code/SeqTGI-IoTID/confusion_allrealdevice.npy', confusion_mat)

