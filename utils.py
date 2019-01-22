#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 16:07:52
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os,json
import logging, sys, argparse
import numpy as np
# np.set_printoptions(threshold=np.inf)
import pandas as pd

def get_logger(filename):
	logger = logging.getLogger('logger')
	logger.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(message)s', level=logging.DEBUG)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)
	return logger

def load_data(path,mode='train'):
	# with open(file=path,mode='r',encoding='utf-8') as file:
	data_frame =  pd.read_csv(path, header=0)
	header = data_frame.columns
	dataset = data_frame.values

	data_x = dataset[:, 1:33]#2-33列为特征

	dataX_dis = feature_discreze(data_x)
	dataX = feature_normalize(dataX_dis)

	whole_data = dataX
	if mode=='train':
		label = dataset[:,33]
		whole_data = (dataX,label)
		pass
	return whole_data
	pass

def feature_discreze(rawdata):
	#将原始数据离散化、归一化
	config_path = os.path.join(os.path.dirname(__file__), './common/normalize.json')
	with open(config_path,'r') as f:
		filecontent = json.load(f)
		config = filecontent['discreter']

	normalize_column = [2,3,7,8,10,11,14,16,19,21,23,24,26,27,28,31]
	num_onehot = 32-len(normalize_column)
	for x in normalize_column:
		num_onehot += len(config['col_{}'.format(x+1)])
		pass
	# print(num_onehot) #处理之后X的维度
	data = np.zeros(shape=(len(rawdata), num_onehot), dtype=float)

	for i,line in enumerate(rawdata):  #离散化
		k = 0
		for j,char in enumerate(line):
			if j in normalize_column:
				normal_list = config['col_{}'.format(j+1)]
				data[i][k + normal_list.index(char)] = 1
				k += len(normal_list)
				pass
			else:
				data[i][k] = char
				k += 1
			pass
		pass
	return data
	pass

def feature_normalize(X):
	fields = [0,1,10,11,12,22,37,38,57,68,69,72,75,82,107,108]
	config_path = os.path.join(os.path.dirname(__file__), './common/normalize.json')
	with open(config_path,'r') as f:
		filecontent = json.load(f)
		config = filecontent['normalizer']

	ret = 1.0 * X
	for index in fields:
		ret[:, index] = (X[:, index] - config[str(index)]['mean']) / config[str(index)]['std']
	return ret
	pass

def means_std(X,Y):
	fields = [0,1,10,11,12,22,37,38,57,68,69,72,75,82,107,108]
	print(X.shape,Y.shape)
	total = np.vstack((X,Y))
	print(total.shape)
	dictor = {}
	temp = {}
	for i in fields:
		print("第%d列==========>"%i)
		arr = np.asarray(total[:,i])
		print(arr.shape)
		b = str(i)
		temp['mean'] = arr.mean()
		temp['std'] = arr.std()
		dictor[b] = temp
		print("平均值为：%f" % arr.mean())
		print("方差为：%f" % arr.std())
		pass
	print(dictor)
	pass

def get_next_batch(train_data, batch_size, shuffle=False):
	if shuffle:
		index=np.arange(len(train_data[0]))
		np.random.shuffle(index)
		data_x = np.array(train_data[0])[index]
		data_y = np.array(train_data[1])[index]
	else:
		data_x = np.array(train_data[0])
		data_y = np.array(train_data[1])
	a=[]
	for data in zip(data_x,data_y):
		if len(a) == batch_size:
			yield a
			a=[]
		a.append(data)
		pass
	if len(a) == batch_size:
		yield a
	pass