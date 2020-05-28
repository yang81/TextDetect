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


class Module():
	def __init__(self):
		self.config = Config()

	def terminate(self):
		self.solver.snapshot()

		layers = self.solver.net.layers
		layer_names = self.solver.net._layer_names

		for i in range(len(layers)):
			if layers[i].type == "Python" and layer_names[i] == "input-data":
				layers[i].terminate_data_process()
				break

		exit(0)

	def sig_handler(self, signalNum, handler):
		print "ctrl + c pressed!"
		self.terminate()

	def createDB(self):
		db = CreateLMDB()
		db.write()

	def train(self, restore = False):
		caffe.set_device(0)
		caffe.set_mode_gpu()

		self.solver = caffe.SGDSolver(self.config.solver_file)

		signal.signal(signal.SIGINT, self.sig_handler)

		if restore:
			self.solver.solve(self.config.restore_state)
		else:
			if self.config.copy_weight != "" :
				self.solver.net.copy_from(self.config.copy_weight)
				print 'copy net'

			while self.config.niter != 0:
				self.solver.step(1)
				self.config.niter -= 1

		self.terminate()

	def convertVertex(self, src):
		hx = src[0]
		hy = src[1]
		tx = src[2]
		ty = src[3]
		h = src[4]/2.
		alpha = src[5]

		v1x = hx - h*math.sin(alpha)
		v1y = hy - h*math.cos(alpha)

		v4x = hx + h*math.sin(alpha)
		v4y = hy + h*math.cos(alpha)

		v2x = tx - h*math.sin(alpha)
		v2y = ty - h*math.cos(alpha)

		v3x = tx + h*math.sin(alpha)
		v3y = ty + h*math.cos(alpha)

		return [(v1x, v1y), (v2x, v2y), (v3x, v3y), (v4x, v4y)]

	def test(self):
		caffe.set_device(0)
		caffe.set_mode_gpu()	

		net = caffe.Net(self.config.model, self.config.weights, caffe.TEST)

		img = Image.open(self.config.image)
		
		img_array = np.asarray(img, dtype='f')
		img_array /= 255

		net.blobs['data'].reshape(1, 3, img.height, img.width)
		net.blobs['data'].data[...] = img_array.transpose((2,0,1))
		

		net.forward()
		show = net.blobs['show'].data[0][0]

		draw = ImageDraw.Draw(img)
		for i in range(show.shape[0]):
			vertexes =  self.convertVertex(show[i])
			draw.polygon(vertexes, outline = (255,0,0))

		img.show()

if __name__ == '__main__':
	module = Module()
	#module.createDB()
	module.train(True)
	#module.test()
	




