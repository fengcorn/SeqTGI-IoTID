#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pcap_splitter.splitter import PcapSplitter
import os
import numpy as np
names = ['Smart_Things', 'Amazon_Echo', 'Netatmo_Welcome', 'TP-Link_Day_Night_Cloud_camera', 'Samsung_SmartCam',
        'Dropcam', 'Insteon_Camera', 'Insteon_Camera', 'Withings_Smart_Baby_Monitor', 'Belkin_Wemo_switch',
        'TP-Link_Smart_plug', 'iHome', 'Belkin_wemo_motion_sensor', 'NEST_Protect_smoke_alarm',
        'Netatmo_weather_station', 'Withings_Smart_scale', 'Blipcare_Blood_Pressure_meter',
        'Withings_Aura_smart_sleep_sensor', 'Light_Bulbs_LiFX_Smart_Bulb', 'Triby_Speaker',
        'PIX-STAR_Photo-frame', 'HP_Printer', 'Nest_Dropcam',]
macs = ['d0:52:a8:00:67:5e', '44:65:0d:56:cc:d3', '70:ee:50:18:34:43', 'f4:f2:6d:93:51:f1', '00:16:6c:ab:6b:88',
       '30:8c:fb:2f:e4:b2', '00:62:6e:51:27:2e', 'e8:ab:fa:19:de:4f', '00:24:e4:11:18:a8', 'ec:1a:59:79:f4:89',
       '50:c7:bf:00:56:39', '74:c6:3b:29:d7:1d', 'ec:1a:59:83:28:11', '18:b4:30:25:be:e4', '70:ee:50:03:b8:ac',
       '00:24:e4:1b:6f:96', '74:6a:89:00:2e:25', '00:24:e4:20:28:c6', 'd0:73:d5:01:83:08', '18:b7:9e:02:20:44',
       'e0:76:d0:33:bb:85', '70:5a:0f:e4:9b:c0', '30:8c:fb:b6:ea:45']

#!rm unsw/images_Blipcare_Blood_Pressure_meter.npy
#!rm unsw/images_Belkin_wemo_motion_sensor.npy
#!rm unsw/images_Dropcam.npy

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


def pcap_split(pcap_file, output_path, mac):
    ps = PcapSplitter(pcap_file)
    print(ps.split_by_session(output_path, pkts_bpf_filter="ether host {}".format(mac)))


def split_pcap():
    source_path = '/home/fengcorn/datasets/captures_unsw/'
    pcaps = os.listdir(source_path)
    for pcap in pcaps:
        if pcap.endswith('.pcap'):
            p = source_path + pcap
            for idx, device_mac in enumerate(macs):
                if not os.path.exists('/home/fengcorn/datasets/captures_unsw/' + names[idx]):
                    os.mkdir('/home/fengcorn/datasets/captures_unsw/' + names[idx])
                pcap_split(p, '/home/fengcorn/datasets/captures_unsw/' + names[idx], device_mac)


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
                image = b''.join(data)[:300*12]
                image = image.ljust(300*12, b'\x00')
                image = np.array(list(image))
                images.append(image)
                print(pcaps_path + ' {}'.format(len(images)))
                if len(images) >= 10000:
                    break
    images = np.array(images)
    np.save(pcaps_path + '/images.npy', images)


def all_pcaps2npy():
    path = '/home/fengcorn/datasets/captures_unsw/'
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
    path = '/home/fengcorn/datasets/captures_unsw/'
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
            data = data.reshape([data.shape[0], 60, 60])
            print(data.shape)
            if len(data) > 1:
                label_name.append(device_path)
                while len(data) < 1000:
                    data = augment(data)
                if data.shape[0] % 5 != 0:
                    idx = data.shape[0] - data.shape[0] % 5
                    print(idx)
                    data = data[:idx]
                data = data.reshape([data.shape[0]//5, 5, 60, 60])
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
    x = x.reshape([x.shape[0], 5, 60, 60])
    y = np.array(y)
    print('{} types'.format(len(label_name)))
    print('total counts: {}'.format(len(y)))
    print('x_shape: {}'.format(x.shape))
    np.save('./train_unsw_60_60.npy', x)
    np.save('./label_unsw_60_60.npy', y)

def statistics():
    import scapy.all
    path = '/home/fengcorn/datasets/captures_unsw/'
    devices_path = os.listdir(path)
    stats = {}
    print(devices_path)
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