#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-16 16:07:52
# @Author  : Liu Lang (liu_lang01@163.com)
# @Link    : ${link}
# @Version : $Id$

import os
import logging, sys, argparse
import numpy as np

def get_logger(filename):
	logger = logging.getLogger('logger')
	logger.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(message)s', level=logging.DEBUG)
	handler = logging.FileHandler(filename)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)
	return logger

def load_data(path):

	pass