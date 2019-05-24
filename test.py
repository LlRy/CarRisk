#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 20:13:39
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os,sys,argparse,imp
import numpy as np
import tensorflow as tf
from utils import load_data
from utils import get_next_batch
from utils import get_logger

def get_feed_dict(batch_sample):
	rawx,rawy = [],[]
	for x in batch_sample:
		rawx.append(x[0])
		rawy.append(x[1])
		pass
	feed_dict = {x: rawx,y: rawy}
	return feed_dict
	pass


datapath = './data/'

train_data_path = os.path.join(datapath,'train.csv')
train_data = load_data(train_data_path,mode='train')
train_part = (train_data[0][8000:],train_data[1][8000:])
val_part = (train_data[0][0:8000],train_data[1][0:8000])

x = tf.placeholder(tf.float32, shape=[None, 111], name="inputX")
y = tf.placeholder(tf.float32, shape=[None], name="inputY")

with tf.variable_scope("layer1"):
	W1 = tf.get_variable(name="W1",
	                    shape=[111, 128],
	                    initializer=tf.contrib.layers.xavier_initializer(),
	                    dtype=tf.float32,
	                    trainable=True)
	b1 = tf.get_variable(name="b1",
	                    shape=[128],
	                    initializer=tf.zeros_initializer(),
	                    dtype=tf.float32,
	                    trainable=True)
	wx_plus_b1=tf.matmul(x, W1) + b1
	middle = tf.nn.softmax(wx_plus_b1)  ##中间层激活
	middle = tf.nn.dropout(middle, 0.5)

with tf.variable_scope("layer2"):
	W2 = tf.get_variable(name="W2",
	                    shape=[128, 1],
	                    initializer=tf.contrib.layers.xavier_initializer(),
	                    dtype=tf.float32,
	                    trainable=True)
	b2 = tf.get_variable(name="b2",
	                    shape=[1],
	                    initializer=tf.zeros_initializer(),
	                    dtype=tf.float32,
	                    trainable=True)
	wx_plus_b2=tf.matmul(middle, W2) + b2
# predict = tf.nn.softmax(wx_plus_b2)

#损失函数选用RMSE
h = (y * tf.log(tf.sigmoid(wx_plus_b2)) + (1-y)* tf.log(1-tf.sigmoid(wx_plus_b2))) * -1
loss = tf.reduce_mean(h)

# loss = tf.sqrt(tf.reduce_mean(tf.square(y - predict)))
#优化函数选取梯度下降法
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	num_batches = len(train_part[0]) // 20
	for epoch in range(20):
		print("=======>epoch %d"%epoch)
		batches = get_next_batch(train_part, 20, shuffle=False)
		for step, batchsample in enumerate(batches):
			step_num = epoch * num_batches + step + 1
			rawx,rawy = [],[]
			for t in batchsample:
				rawx.append(t[0])
				rawy.append(t[1])
				pass
			feed_dict = {x: rawx, y: rawy}
			# feed_dict = get_feed_dict(batchsample)#整理喂入神经网络的字典
			# feed_dict = {x: batch_rawx, y: batch_rawy}
			loss_train, _ = sess.run([loss,train_op], feed_dict = feed_dict)
			if step + 1 == 1 or (step + 1) % 200 == 0 or step + 1 == num_batches:
			    print('epoch {}, step {}, loss: {:.4}, global_step: {}'.format(
			                                                                    epoch,
			                                                                    step + 1,
			                                                                    loss_train, step_num))
		pass