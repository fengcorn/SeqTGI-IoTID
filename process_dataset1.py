#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from pcap_splitter.splitter import PcapSplitter
import os
import numpy as np


def erase_mac(raw):
    if type(raw) != bytes:
        raw = raw.original
    new = raw[16:]
    new = b'\x00' * 16 + new
    return new

def erase_ip(raw):
    if type(raw) != bytes:
        raw = raw.original
    new = raw[:27] + b'\x00' * 8 + raw[35:]
    return new


def pcap_split(pcap_file, output_path):
    ps = PcapSplitter(pcap_file)
    print(ps.split_by_session(output_path))


def split_all_files():
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    devices_path = os.listdir(path)
    for device_path in devices_path:
        if os.path.isdir(path + device_path):
            pcaps = os.listdir(path + device_path)
            if not os.path.exists(path + device_path + '/sessions'):
                os.mkdir(path + device_path + '/sessions')
            for pcap in pcaps:
                if pcap.endswith('.pcap'):
                    ps = PcapSplitter(path + device_path + '/' + pcap)
                    print(ps.split_by_session(path + device_path + '/sessions'))


def view_images():
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    devices_path = os.listdir(path)
    for device_path in devices_path:
        if os.path.isdir(path + device_path):
            data = np.load(path + device_path + '/sessions/' + 'images.npy')
            print(device_path, data)
            cv2.imwrite(device_path + '.jpg', data[8].reshape([28, 28]))


def has_raw(packets):
    if 1 in [p.haslayer('Raw') for p in packets]:
        return 1
    else:
        return 0


def has_ip(packets):
    if 1 in [p.haslayer('IP') for p in packets]:
        return 1
    else:
        return 0


def pcaps2npy(pcaps_path):
    import scapy.all
    pcaps = os.listdir(pcaps_path)
    pcaps = sorted(pcaps)
    images = []
    for pcap in pcaps:
        if pcap.endswith('.pcap'):
            packets = scapy.all.rdpcap(pcaps_path + '/' + pcap)
            if has_raw(packets):
                # print(pcap)
                data = []
                for p in packets:
                    tmp = p.original
                    if 'IP' in p:
                        tmp = erase_ip(tmp)
                    data.append(tmp)
                data = [erase_mac(d) for d in data]
                image = b''.join(data)[:784]
                image = image.ljust(784, b'\x00')
                image = np.array(list(image))
                images.append(image)
    images = np.array(images)
    np.save(pcaps_path + '/images.npy', images)


def all_pcaps2npy():
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    devices_path = os.listdir(path)
    devices_path = sorted(devices_path)
    for device_path in devices_path:
        print(device_path)

        if os.path.isdir(path + device_path):
            pcaps = path + device_path + '/sessions'
            pcaps2npy(pcaps)
            # break

# 数据增强，让各个类型设备数据保持平衡。
def augment(data):
    new_data = np.copy(data)
    noise = (np.random.normal(scale=1/255, size=data.shape)*255).astype(int)
    tmp = np.clip(data + noise, 0, 255)
    new_data = np.concatenate([new_data, tmp])
    noise = (np.random.normal(scale=2/255, size=data.shape)*255).astype(int)
    tmp = np.clip(data + noise, 0, 255)
    new_data = np.concatenate([new_data, tmp])
    noise = (np.random.normal(scale=3/255, size=data.shape)*255).astype(int)
    tmp = np.clip(data + noise, 0, 255)
    new_data = np.concatenate([new_data, tmp])
    return new_data

# 拼接成一个大的npy文件，用于训练模型。
def remove_and_concat():
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    devices_path = os.listdir(path)
    devices_path = sorted(devices_path)
    print(devices_path)
    x = []
    y = []
    x_label = 0
    label_name = []
    for device_path in devices_path:
        if os.path.isdir(path + device_path):
            data = np.load(path + device_path + '/sessions/' + 'images.npy')
            #data = np.load('/home/fengcorn/datasets/captures_IoT-Sentinel/' + 'Aria' + '/sessions/' + 'images.npy')
            data = data.reshape([data.shape[0], 28, 28])
            print(data.shape)
            if len(data) > 1:
                label_name.append('Aria')
                #while len(data) < 300:
                #    data = augment(data)
                #while len(data) > 1000:
                #    data = data[:len(data)//2]
                if data.shape[0] % 5 != 0:
                    idx = data.shape[0] - data.shape[0] % 5
                    print(idx)
                    data = data[:idx]
                data = data.reshape([data.shape[0]//5, 5, 28, 28])
                print(data.shape)
                y += [x_label] * len(data)
                x.extend(data)
                x_label += 1
                if device_path in ['EdimaxCam1', 'EdnetCam1', 'WeMoInsightSwitch', 'WeMoSwitch']:
                    x_label -= 1
    import collections
    data_count2 = collections.Counter(np.array(y))
    print(label_name)
    print(data_count2)
    x = np.array(x)
    x = x.reshape([x.shape[0], 5, 28, 28])
    y = np.array(y)
    print('{} types'.format(len(label_name)))
    print('total counts: {}'.format(len(y)))
    print(x.shape)
    #np.save('./train_n12.npy', x)
    #np.save('./label_n12.npy', y)
    np.save('./train12.npy', x)
    np.save('./label12.npy', y)

def split_train_and_test(x, y, test_size):
    x_train, x_test, y_train, y_test = [], [], [], []
    for i in range(y.max()+1):
        all_x = x[y==i]
        shuf = np.random.permutation(len(all_x))
        all_x = all_x[shuf]
        percent = int(len(all_x)*(1-test_size))
        x_train.extend(all_x[:percent])
        x_test.extend(all_x[percent:])
        y_train.extend([i]*percent)
        y_test.extend([i]*(len(all_x)-percent))
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test

def statistics():
    import scapy.all
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    devices_path = os.listdir(path)
    stats = {}
    for device_path in devices_path:
        print(device_path)
        if os.path.isdir(path + device_path):
            stat = {}
            pcaps = os.listdir(path + device_path + '/sessions')
            for pcap in pcaps:
                if pcap.endswith('.pcap'):
                    packets = scapy.all.rdpcap(path + device_path + '/sessions/' + pcap)
                    if len(packets) in stat:
                        stat[len(packets)] += 1
                    else:
                        stat[len(packets)] = 1
            stats[device_path] = stat
    print(stats)

def print_labels():
    path = '/home/fengcorn/datasets/captures_IoT-Sentinel/'
    print(sorted(os.listdir(path)))

if __name__ == '__main__':
    print_labels()
    split_all_files()
    all_pcaps2npy()
    remove_and_concat()
    # view_images()
    # statistics()