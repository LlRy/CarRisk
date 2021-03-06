#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 16:08:26
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os,time,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from utils import get_next_batch

class Network(object):
	def __init__(self, args, num_classes=1, input_dim=111, **kwargs):
		self.input_dim = input_dim
		self.depth = 2
		self.hidden_dim = 128
		self.task = 'regression'
		self.batch_size = 20
		self.dropout_pl = 0.5
		self.lr = 0.01
		self.clip_grad = 5.0
		self.shuffle = False
		self.output_dim = 1
		self.network_op()
		pass

	def init_model_save_path(self,paths):
		self.model_path = os.path.join(paths['model_path'], "mymodel")  #给模型加前缀
		self.summary_path = paths['summary_path']
		pass

	def addSummary(self, sess):
		"""
		:param sess:
		:return:
		"""
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

	def network_op(self):
		self.inputX = tf.placeholder(tf.float32, shape=[None, self.input_dim], name="inputX")
		self.inputY = tf.placeholder(tf.float32, shape=[None], name="inputY")

		with tf.variable_scope("layer1"):
			W1 = tf.get_variable(name="W1",
                                shape=[self.input_dim, self.hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			b1 = tf.get_variable(name="b1",
                                shape=[self.hidden_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			wx_plus_b1=tf.matmul(self.inputX, W1)+b1
			middle = wx_plus_b1
			# middle = tf.nn.softmax(wx_plus_b1)  ##中间层激活
			# middle = tf.nn.dropout(middle,self.dropout_pl)

		with tf.variable_scope("layer2"):
			W2 = tf.get_variable(name="W2",
                                shape=[self.hidden_dim, self.output_dim],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			b2 = tf.get_variable(name="b2",
                                shape=[self.output_dim],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                trainable=True)
			wx_plus_b2=tf.matmul(middle,W2)+b2

		# self.logits = tf.reshape(pred, [-1])
		self.predict = tf.reshape(wx_plus_b2,[-1])
		self.predict_score_ = tf.sigmoid(self.predict)
		# self.predict=tf.nn.softmax(wx_plus_b2)

		#损失函数选用RMSE
		# self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.inputY-self.predict)))
		self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.inputY, self.predict))))

		# self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputY,logits=wx_plus_b2)
		# h = (self.inputY * tf.log(tf.sigmoid(wx_plus_b2)) + (1-self.inputY)* tf.log(1-tf.sigmoid(wx_plus_b2))) * -1
		# self.loss = tf.reduce_mean(h)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9)
		grads_and_vars = optim.compute_gradients(self.loss)
		grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
		self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
		# self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
		# AdamOptimizer(self.lr,0.9).minimize(self.loss)
		# GradientDescentOptimizer(self.lr).minimize(self.loss)
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
		num_batches = len(train_data[0]) // self.batch_size
		start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

		batches = get_next_batch(train_data, self.batch_size, shuffle=self.shuffle)
		for step, batchsample in enumerate(batches):
			step_num = (epoch-1) * num_batches + step + 1
			feed_dict, _ = self.get_feed_dict(batchsample,self.lr,self.dropout_pl)#整理喂入神经网络的字典
			_, loss_train = sess.run([self.train_op, self.loss],feed_dict=feed_dict)
			if step + 1 == 1 or (step + 1) % 200 == 0 or step + 1 == num_batches:
			    logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(
			                                                                    start_time,
			                                                                    epoch,
			                                                                    step + 1,
			                                                                    loss_train, step_num))

			# self.file_writer.add_summary(summary, step_num)
		saver.save(sess, self.model_path, global_step = epoch)
		pass

	def predict_one_epoch(self, sess, validate):
		feed_dict = {self.inputX: validate[0],self.inputY: validate[1]}
		predictions, losses = sess.run([self.predict_score_, self.loss], feed_dict=feed_dict)
		return predictions, validate[1], losses
		pass

	def get_feed_dict(self, batch_sample, lr=None, dropout=None):
		rawx,rawy = [],[]
		for x in batch_sample:
			rawx.append(x[0])
			rawy.append(x[1])
			pass
		feed_dict = {self.inputX: rawx,
		             self.inputY: rawy}
		return feed_dict, rawy
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