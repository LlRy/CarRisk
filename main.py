#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 16:07:23
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os,sys,argparse,imp
import numpy as np
import tensorflow as tf
from utils import load_data
from utils import get_logger

#添加程序参数
parser = argparse.ArgumentParser(description='LSTM-based deep learning methods for MIMIC-III clinical database')
parser.add_argument('--network', type=str, default="./models/SVM.py", help='model structure')
parser.add_argument('--data', type=str, default='./data/', 
                    help='train data source')
parser.add_argument('--epoch',type=int,default=20,help='number of epochs')
parser.add_argument('--mode',type=str,default="train",help="train/test")
parser.add_argument('--load_state',type=str,default='',help='model checkpoint path')
args = parser.parse_args()

'''
定义模型，初始化计算图
'''

assert args.network is not None
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(args, num_classes = 1, input_dim = 111)
model.final_name = model.say_name()

'''
 设置模型的各种保存路径包括：
 日志记录路径log_path
 最终epoch的模型计算图model_path
 预测的结果保存路径prediction_path
'''
paths = {}
modelname = model.final_name if args.load_state == '' else args.load_state
output_path = os.path.join('.',"output_save_path",modelname)
model_path = os.path.join(output_path, "checkpoints/")
result_path = os.path.join(output_path, "results")
summary_path = os.path.join(output_path, "summaries")
log_path = os.path.join(result_path, "log.txt")
paths['model_path'] = model_path
paths['summary_path'] = summary_path
model.init_model_save_path(paths)
if not os.path.exists(result_path): 
    os.makedirs(result_path)
    pass
logger = get_logger(log_path)  #设置日志记录器
if args.load_state == "":
    logger.info(str(args))
    pass

'''
判断模型是否重载
如果是模型重新载入，需要读出计算图的路径，以及最新的迭代次数
ckpt_file、global_epoch
'''
ckpt_file = None
if args.load_state != "":
    if not os.path.exists(model_path):
        raise Exception("the folder {} is not exists".format(model_path))
    ckpt_file = tf.train.latest_checkpoint(model_path)
global_epoch = 0
if not ckpt_file == None: 
    global_epoch = int(re.findall(r'\d+', ckpt_file)[-1])
print("==>At model epoch",global_epoch)

'''
根据self.mode进入相应的控制程序
'''

if args.mode == 'train':
	###读取训练和验证数据
	train_data_path = os.path.join(args.data,'train.csv')
	# val_data_path = os.path.join(args.data,'test.csv')
	train_data = load_data(train_data_path,mode='train')
	# val_data = load_data(val_data_path,mode='test')
	# print("training mode")
	# exit()
	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		#如果模型有checkpoint,载入
		if ckpt_file == None:
			sess.run(tf.global_variables_initializer())
			pass
		else: #如果存在checkpoint，先加载
			saver.restore(sess, ckpt_file)
		#逐个epoch进行训练
		for epoch in range(args.epoch):
			#分
			logger.info('===========training on epoch {}==========='.format(epoch+global_epoch+1))
			model.train_epoch(sess, train_data, epoch+global_epoch+1, saver, logger)
			logger.info('==> loss on train dataset')
			predictions,labels,losses = model.predict_one_epoch(sess, train_data)  ##计算验证集的loss
			print("loss on training dataset:",losses)
			# val_result = print_metrics_binary(labels, predictions,verbose=1)
			# val_result['losses'] = losses
			# logger.info(val_result)
			
			pass
		pass
	pass
elif args.mode == 'test':
	'''
	读取测试数据
	'''
	test_data_path = os.path.join(args.data,'test.csv')
	test_data = load_data(test_data_path)
	if ckpt_file ==None:
		raise Exception("No checkpoint model was found")
		pass
	with tf.Session() as sess:

		pass
	pass #args.mode==test结束
else:
	raise ValueError("Wrong value [{}] for parameter args.mode".format(args.mode))