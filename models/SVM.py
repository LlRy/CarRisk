#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 16:08:26
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os
import tensorflow as tf

class Network(object):
	def __init__(self, args, num_classes=1, input_dim=32, **kwargs):
		self.input_dim = input_dim
		self.depth = 
		self.hidden_dim = 
		self.task = 'regression'
		self.batch_size = 20
		self.dropout_pl = 0.5
		self.lr = 0.001
		self.output_dim = 1
		self.network_op()
		pass

	def network_op(self):
		self.inputX = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="inputX")
		self.inputY = tf.placeholder(tf.float32, shape=[None, self.output_dim], name="inputY")

		with tf.variable_scope("layer1"):
			W1 = tf.get_variable(name="W",
                                shape=[self.input_dim, self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32,
                                trainable=True)
            b1 = tf.get_variable(name="b",
                                shape=[self.hidden_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			wx_plus_b1=tf.matmul(self.inputX,W1)+b1
			middle = tf.nn.softmax(wx_plus_b1)  ##中间层激活
			middle = tf.nn.dropout(middle,self.dropout_pl)

		with tf.variable_scope("layer2"):
			W2 = tf.get_variable(name="W",
                                shape=[self.hidden_dim, self.output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32,
                                trainable=True)
            b2 = tf.get_variable(name="b",
                                shape=[self.output_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			wx_plus_b2=tf.matmul(middle,W2)+b2
			self.predict=tf.nn.softmax(wx_plus_b2)

		#损失函数选用RMSE
		loss=tf.sqrt(tf.reduce_mean(tf.square(self.inputY-self.predict)))
		#优化函数选取梯度下降法
		train=tf.train.GradientDescentOptimizer(self.lr).minimize(loss)
		pass

	def train_epoch(self, sess, train_data, epoch, saver, logger):
		'''模型训练一个epoch的函数
		Arguments:
			sess {[type]} -- [description]
			train_data {[type]} -- [description]
			epoch {[type]} -- [description]
			saver {[type]} -- [description]
			logger {[type]} -- [description]
		'''
		pass
	def say_name(self):
		return "svm_n{}_dep{}_drop{}_lr{}_bat{}_t{}".format(self.hidden_dim,
                                                            self.depth,
                                                            self.dropout_pl,
                                                            self.lr,
                                                            self.batch_size,
                                                            str(int(time.time()))
                                                            )
		pass