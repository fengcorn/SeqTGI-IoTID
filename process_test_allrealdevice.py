#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/18 10:42 AM
# @Author  : Eustiar
# @File    : process_unsw.py
from pcap_splitter.splitter import PcapSplitter
import os
import numpy as np
import random

names = ['NodeMCU_MQTT', 'NodeMCU_HTTPS', 'NodeMCU_CoAP', 'NodeMCU_SMTP', 'NodeMCU_NTP', 'NodeMCU_FTP', '360Camera', 'MiCamera', 'SmartPlug_1', 'SmartPlug_2']
ip_address = ['192.168.137.96', '192.168.137.185', '192.168.137.18', '192.168.137.108', '192.168.137.251', '192.168.137.163', '192.168.137.67', '192.168.137.134', '192.168.137.170', '192.168.137.79']

def erase_mac(raw):
    if type(raw) != bytes:
        raw = raw.original
    new = raw[16:]
    new = b'\x00' * 16 + new
    # new = new.rjust(300, b'\x00')
    return new

def erase_ip(raw):
    if type(raw) != bytes:
        raw = raw.original
    new = raw[:27] + b'\x00' * 8 + raw[35:]
    return new


def has_ip(packets):
    if 1 in [p.haslayer('IP') for p in packets]:
        return 1
    else:
        return 0

def pcap_split(pcap_file, output_path, ip_address):
    ps = PcapSplitter(pcap_file)
    print(ps.split_by_session(output_path, pkts_bpf_filter="ip host {}".format(ip_address)))

def pcap_split_random(pcap_file, output_path, ip_address):
    ps = PcapSplitter(pcap_file)
    packet_count = random.randint(10, 20)  # Generate a random number between 10 and 20
    print(ps.split_by_count(packet_count, output_path, pkts_bpf_filter="ip host {}".format(ip_address)))

def split_pcap():
    source_path = '/home/fengcorn/datasets/test_allrealdevice/'
    pcaps = os.listdir(source_path)
    for pcap in pcaps:
        if pcap.endswith('.pcap'):
            p = source_path + pcap
            for idx, device_ip_address in enumerate(ip_address):
                if not os.path.exists('/home/fengcorn/datasets/test_allrealdevice/' + names[idx]):
                    os.mkdir('/home/fengcorn/datasets/test_allrealdevice/' + names[idx])
                pcap_split_random(p, '/home/fengcorn/datasets/test_allrealdevice/' + names[idx], device_ip_address)


def pcaps2npy(pcaps_path):
    import scapy.all
    pcaps = os.listdir(pcaps_path)
    pcaps = sorted(pcaps)
    images = []
    for pcap in pcaps:
        if pcap.endswith('.pcap'):
            packets = scapy.all.rdpcap(pcaps_path + '/' + pcap)
            if 1:  # has_raw(packets):
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
                print(pcaps_path + ' {}'.format(len(images)))
                if len(images) >= 10000:
                    break
    images = np.array(images)
    np.save(pcaps_path + '/images.npy', images)


def all_pcaps2npy():
    path = '/home/fengcorn/datasets/test_allrealdevice/'
    devices_path = os.listdir(path)
    devices_path = sorted(devices_path)
    for device_path in devices_path:
        print(device_path)
        if os.path.isdir(path + device_path):
            pcaps = path + device_path
            pcaps2npy(pcaps)

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

def remove_and_concat():
    path = '/home/fengcorn/datasets/test_allrealdevice/'
    devices_path = os.listdir(path)
    devices_path = sorted(devices_path)
    x = []
    y = []
    x_label = 0
    label_name = []
    for device_path in devices_path:
        if os.path.isdir(path + device_path):
            print(device_path)
            data = np.load(path + device_path + '/images.npy')
            data = data.reshape([data.shape[0], 28, 28])
            print(data.shape)
            if len(data) > 1:
                label_name.append(device_path)
                while len(data) < 1000:
                    data = augment(data)
                if data.shape[0] % 5 != 0:
                    idx = data.shape[0] - data.shape[0] % 5
                    print(idx)
                    data = data[:idx]
                data = data.reshape([data.shape[0]//5, 5, 28, 28])
                print(len(data))
                y += [x_label] * len(data)
                x.extend(data)
                x_label += 1
                print(len(x))
                print(len(y)) 
    import collections
    data_count2 = collections.Counter(np.array(y))
    print(label_name)
    print(data_count2)
    x = np.array(x)
    x = x.reshape([x.shape[0], 5, 28, 28])
    y = np.array(y)
    print('{} types'.format(len(label_name)))
    print('total counts: {}'.format(len(y)))
    print('x_shape: {}'.format(x.shape))
    np.save('./train_test_allrealdevice_28_28.npy', x)
    np.save('./label_test_allrealdevice_28_28.npy', y)

def statistics():
    import scapy.all
    path = '/home/fengcorn/datasets/test_allrealdevice/'
    devices_path = os.listdir(path)
    stats = {}
    print(devices_path)
    # device_path = 'Amazon_Echo'
    for device_path in devices_path:
        print(device_path)
        if os.path.isdir(path + device_path):
            stat = {}
            stat_p = {}
            pcaps = sorted(os.listdir(path + device_path))
            for pcap in pcaps:
                print(pcap)
                if pcap.endswith('.pcap'):
                    packets = scapy.all.rdpcap(path + device_path + '/' + pcap)
                    if len(packets) in stat:
                        stat[len(packets)] += 1
                    else:
                        stat[len(packets)] = 1
                    for p in packets:
                        if len(p) in stat_p:
                            stat_p[len(p)] += 1
                        else:
                            stat_p[len(p)] = 1
            stats[device_path] = stat
    print(stats)       

if __name__ == '__main__':
    split_pcap()
    all_pcaps2npy()
    remove_and_concat()
    statistics()