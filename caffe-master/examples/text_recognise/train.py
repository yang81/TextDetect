#!/usr/bin/env python
# -*- coding: utf-8 -*-
__metaclss__=type

import os
import sys
sys.path.append(os.getcwd() + "/python")
import caffe
from utils.db import CreateLMDB


import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from PIL import Image, ImageDraw, ImageFont
import signal
from utils.config import Config
from utils.sample import WordList


class Module():
	def __init__(self):
		self.config = Config()
		self.word_list = WordList(self.config)

	def terminate(self):
		self.solver.snapshot()
		exit(0)

	def sig_handler(self, signalNum, handler):
		print "ctrl + c pressed!"
		self.terminate()

	def print_weights(self, name, p = False):
		if p :
			print self.solver.net.params[name][0].data

	def print_diff(self, name, p = False):
		if p :
			print self.solver.net.params[name][0].diff

	def show_data(self, name, channels):

		for i in range(channels):
			data = self.solver.net.blobs[name].data[0, i]
			print data.shape
			fig = plt.figure()
			fig.canvas.manager.window.move(300,300)
			plt.imshow(data, cmap='gray')
			plt.show()

	def train(self, restore = False):
		caffe.set_device(0)
		caffe.set_mode_gpu()

		if self.config.recognise_solver_type == "SGD":
			self.solver = caffe.SGDSolver(self.config.recognise_solver_file)
		elif self.config.recognise_solver_type == "Adam":
			self.solver = caffe.AdamSolver(self.config.recognise_solver_file)
		else:
			print "unknow solver type"
			
		signal.signal(signal.SIGINT, self.sig_handler)

		if restore:
			self.solver.solve(self.config.recognise_restore_state)
		else:
			if self.config.recognise_copy_weight != "" :
				self.solver.net.copy_from(self.config.recognise_copy_weight)
				print 'copy net'

			while self.config.recognise_niter != 0:
				#self.print_weights('lstm2', True)
				self.solver.step(1)
				#self.print_diff('lstm2', True)
				#self.show_data('conv4_1', 64)
				self.config.recognise_niter -= 1

		self.terminate()


	def test(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()

		net = caffe.Net(self.config.recognise_model, self.config.recognise_weights, caffe.TEST)

		img = Image.open(self.config.recognise_image)
		
		img_array = np.asarray(img, dtype='f')
		img_array /= 255

		net.blobs['data'].reshape(1, 1, img.height, img.width)
		net.blobs['data'].data[...] = img_array
		

		net.forward()
		labels = net.blobs['result'].data
		
		print self.word_list.get_array(labels[0, : , 0,0].astype('int'))



if __name__ == '__main__':
	module = Module()

	if sys.argv[1] == "train":
		module.train()
	elif sys.argv[1] == "test":
		module.test()
	else:
		print "unknow"
	




