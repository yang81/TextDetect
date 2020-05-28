#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
import math
import sys
import os
import numpy as np
import time
import collections
import lmdb
from multiprocessing import Process, Queue
import signal
from utils.config import Config
from utils.font import FontObj
from utils.sample import WordList, Generator, ImageSample
import caffe


class OnlineDataLayer(caffe.Layer):

	def __del__(self):
		self.db.close()

	def setup(self, bottom, top):
		print "OnlineDataLayer, setup"

		self.config = Config()
		self.word_list = WordList(self.config)
		self.generator = Generator(self.word_list)

		self.batch = int(1)
		self.height = int(512)
		self.width = int(512)

		self.db = lmdb.open(self.config.db_path)
		self.txn = self.db.begin()
		self.cur = self.txn.cursor()
		self.cur.first()

		self.stop_data_process = False
		self.q = Queue(16)
		self.data_process = Process(target=self.get_data, args=(self.q,))
		self.data_process.start()
	

	def get_data(self, q):
		while not self.stop_data_process:
			img_str = self.cur.value()
			img = Image.frombytes("RGB", (512,512), img_str)
			q.put(img)
			if not self.cur.next():
				self.cur.first()

	def terminate_data_process(self):
		self.stop_data_process = True
		self.data_process.terminate()
		self.data_process.join()
		self.db.close()

	def reshape(self, bottom, top):
		top[0].reshape(self.batch, 3, self.height, self.width)
		#top[1].reshape(1, 1, 1, 5)
		top[1].reshape(1, 1, 1, 9)

	def forward(self, bottom, top):
		img = None

		if random.randint(0,1) == 0:
			img = self.q.get()
		
		sample = ImageSample(self.config, self.generator, img)

		#img,labels = sample.traverseSingleTxt(True, True)
		chs_eng = [True, False]
		img,labels = sample.traverseRandomLenTxt(random.randint(0,10) != 1)
		#img,labels = sample.traverseRandomLenTxt(True)
		
		
		#img,labels = sample.randomSingleWordTxt(True, True)
		#img,labels = sample.randomLenTxt(chs_eng[random.randint(0,1)])
		#img,labels = sample.randomLenTxt(True)

		#self.index += 1
		#self.index %= self.original_count

		top[0].data[...] = img
		top[1].reshape(1, 1, len(labels), 5)
		for i in range(len(labels)):
			top[1].data[0,0,i] = labels[i]

		#print 'ground true x,y,w,h: ', labels[0][0], labels[0][1], labels[0][2], labels[0][3]
		#print 'ground true angle: ', labels[0][4]

	def backward(self, top, propagate_down, bottom):
		pass

class FinalLayer(caffe.Layer):

	def setup(self, bottom, top):
		print "FinalLayer, setup"
		self.score_threshold = 0.7
		self.tcbp_threshold = 0.4
		self.pixel_distance = 100
		self.regions = []

	def reshape(self, bottom, top):
		top[0].reshape(1, 1, 1, 1)

	def __getCenterH(self, x, y, d1, d3, d4, region, alpha = 0.):
		n1 = (d1 + d3)/2. - d1
		n2 = d4

		region['hx'] += x
		region['hy'] += y
		region['hxn1'] += n1
		region['hxn2'] += -n2
		region['hyn1'] += n1
		region['hyn2'] += n2

	def __getCenterT(self, x, y, d1, d3, d2, region, alpha = 0.):
		n1 = (d1 + d3)/2. - d1
		n2 = d2

		region['tx'] += x
		region['ty'] += y
		region['txn1'] += n1
		region['txn2'] += n2
		region['tyn1'] += n1
		region['tyn2'] += -n2


	#region growing segment
	def __rgs(self,i, j, fs, head, tail, label, geometry, angle, width, height, region):
		#boundary check
		if i < 0 or j < 0 or i >= height or j >= width or int(label[i][j]) == 1:
			return

		#set label to 1 (means checked)
		if fs[i][j] > 0:
			label[i][j] = 1

			d1 = geometry[0][i][j]
			d2 = geometry[1][i][j]
			d3 = geometry[2][i][j]
			d4 = geometry[3][i][j]

			alpha = angle[i][j]

			region['angle'] += alpha
			region['count'] += 1
			region['h'] += d1 + d3

			if head[i][j] > 0:
				#get left center
				self.__getCenterH(j*4., i*4., d1, d3, d4, region, alpha)
				region['hcount'] += 1

			if tail[i][j] > 0:
				#get right center
				self.__getCenterT(j*4., i*4., d1, d3, d2, region, alpha)
				region['tcount'] += 1

			#8 connected neighbor pos
			#top
			self.__rgs(i-1, j, fs, head, tail, label, geometry, angle, width, height, region)
			#top left
			self.__rgs(i-1, j-1, fs, head, tail, label, geometry, angle, width, height, region)
			#top right
			self.__rgs(i-1, j+1, fs, head, tail, label, geometry, angle, width, height, region)
			#left
			self.__rgs(i, j-1, fs, head, tail, label, geometry, angle, width, height, region)
			#right
			self.__rgs(i, j+1, fs, head, tail, label, geometry, angle, width, height, region)
			#bottom
			self.__rgs(i+1, j, fs, head, tail, label, geometry, angle, width, height, region)
			#bottom left
			self.__rgs(i+1, j-1, fs, head, tail, label, geometry, angle, width, height, region)
			#bottom right
			self.__rgs(i+1, j+1, fs, head, tail, label, geometry, angle, width, height, region)

		else:
			#score less than or equal to 0
			return


		

	def forward(self, bottom, top):
		score = bottom[0].data[0][0]
		score = np.where(score >= self.score_threshold, score, 0)

		tcbp = bottom[1].data[0][0]
		tcbp = np.where(score > 0, tcbp, 0)
		
		f = np.where(tcbp > self.tcbp_threshold, score, 0)

		geometry = bottom[2].data[0]
		angle = bottom[3].data[0][0]

		gh = np.where(f > 0 , geometry[3], 0)
		head = np.where((gh <= self.pixel_distance) & (gh > 0), 1, 0)

		gt = np.where(f > 0 , geometry[1], 0)
		tail = np.where((gt <= self.pixel_distance) & (gt > 0), 1, 0)

		label = np.zeros(f.shape, dtype=np.int)

		
		i = 0
		j = 0
		for i in range(f.shape[0]):
			for j in range(f.shape[1]):
				if f[i][j] > 0 and label[i][j] == 0: 
					#we get a text region
					region = {'w':0., 'h':0., 'x':0., 'y':0., 'angle':0., 'count':0, 
					'hx':0., 'hy':0., 'hcount':0, 'hxn1':0., 'hyn1':0., 'hxn2':0., 'hyn2':0.,
					'tx':0., 'ty':0., 'tcount':0, 'txn1':0., 'tyn1':0., 'txn2':0., 'tyn2':0. }

					self.__rgs(i, j, f, head, tail, label, geometry, angle, f.shape[1], f.shape[0], region)

					region['h'] = region['h']/region['count']
					region['angle'] = region['angle']/region['count']

					if region['hcount'] == 0:
						region['hcount'] = 1
					if region['tcount'] == 0:
						region['tcount'] = 1
					
					region['hx'] = (region['hx'] +  region['hxn1']*math.sin(region['angle']) +  region['hxn2']*math.cos(region['angle']))/region['hcount']
					region['hy'] = (region['hy'] +  region['hyn1']*math.cos(region['angle']) +  region['hyn2']*math.sin(region['angle']))/region['hcount']
					region['tx'] = (region['tx'] +  region['txn1']*math.sin(region['angle']) +  region['txn2']*math.cos(region['angle']))/region['tcount']
					region['ty'] = (region['ty'] +  region['tyn1']*math.cos(region['angle']) +  region['tyn2']*math.sin(region['angle']))/region['tcount']
					
					region['x'] = (region['hx'] + region['tx'])/2.
					region['y'] = (region['hy'] + region['ty'])/2.
					region['w'] = math.sqrt( (region['hx']-region['tx'])*(region['hx']-region['tx']) +  (region['hy']-region['ty'])*(region['hy']-region['ty']))
					#self.regions.append(region)
					#self.regions.append((region['x'], region['y'], region['w'], region['h'], region['angle']))
					self.regions.append((region['hx'], region['hy'], region['tx'], region['ty'], region['h'], region['angle']))
				j += 1

			i += 1
		

		top[0].reshape(1, 1, len(self.regions), 6)
		for i in range(len(self.regions)):
			top[0].data[0,0,i] = self.regions[i]


	def backward(self, top, propagate_down, bottom):
		pass

class OnlineMSRATD500Layer(caffe.Layer):

	def setup(self, bottom, top):
		print "OnlineMSRATD500Layer, setup"

		self.path = '/root/ocr/data/MSRA-TD500/train'
		self.image_scene_list = []
		self.label_list = []

		self.index = int(0)
		self.batch = int(1)
		self.height = int(512)
		self.width = int(512)

		for root, dirs, files in os.walk(self.path):
			for f in files:
				if 'JPG' in f:
					img = Image.open(os.path.join(root, f))
					self.height = img.height
					self.width = img.width

					self.image_scene_list.append(img)
					f = f.replace('JPG', 'gt')
					with open(os.path.join(root, f), 'r') as fh:
						str_labels = fh.readlines()
						for i in range(len(str_labels)):
							str_labels[i] = str_labels[i].replace('\r\n', '')
							str_labels[i] = str_labels[i].split(' ')
							str_labels[i] = str_labels[i][2:]
							str_labels[i] = [float(x) for x in str_labels[i]]
							str_labels[i][0] += str_labels[i][2]/2
							str_labels[i][1] += str_labels[i][3]/2
						self.label_list.append(str_labels)

		self.original_count = len(self.image_scene_list)

	def reshape(self, bottom, top):
		top[0].reshape(self.batch, 3, self.height, self.width)
		top[1].reshape(1, 1, 1, 5)

	def forward(self, bottom, top):
		
		self.index += 1
		self.index %= self.original_count

		img = self.image_scene_list[self.index]

		img_array = np.asarray(img, dtype='f')
		img_array -= 127.5
		img_array /= 256

		top[0].data[...] = img_array.transpose((2,0,1))
		top[1].reshape(1, 1, len(self.label_list[self.index]), 5)
		for i in range(len(self.label_list[self.index])):
			top[1].data[0,0,i] = self.label_list[self.index][i]
		
		pass

	def backward(self, top, propagate_down, bottom):
		pass

class OnlineTestDataLayer(caffe.Layer):

	def setup(self, bottom, top):
		print "OnlineTestDataLayer, setup"
		path = '/root/ocr/data/test_image/01.png'
		self.img = Image.open(path)
		min_side = min(self.img.height, self.img.width)
		self.img = self.img.resize((512, 512), resample=Image.BILINEAR, box=(0,0, min_side, min_side))
		self.img.show()
		top[0].reshape(1, 3, self.img.height, self.img.width)

		self.index = 785


	def reshape(self, bottom, top):
		top[0].reshape(1, 3, self.img.height, self.img.width)

	def forward(self, bottom, top):
		#path = '/root/ocr/data/image_512_scene_single/0000' + str(self.index) + '.jpg'


		img_array = np.asarray(self.img, dtype='f')
		#img_array -= 127.5
		img_array /= 256

		top[0].data[...] = img_array.transpose((2,0,1))
		self.index += 1

	def backward(self, top, propagate_down, bottom):
		pass


class TestDataLayer(caffe.Layer):

	def setup(self, bottom, top):
		print "TestDataLayer, setup"
		path = '/root/ocr/data/test_image/01.png'
		self.img = Image.open(path)
		self.img.show()

	def reshape(self, bottom, top):
			
		top[0].reshape(1, 3, self.img.height, self.img.width)

	def forward(self, bottom, top):
		top[0].reshape(1, 3, self.img.height, self.img.width)
		
		img_array = np.asarray(self.img, dtype='f')
		img_array /= 256
		top[0].data[...] = img_array.transpose((2,0,1))

	def backward(self, top, propagate_down, bottom):
		pass





if __name__ == '__main__':
	config = Config()
	word_list = WordList(config)
	generator = Generator(word_list)
	generator.getRandomEnglishTxt(32)


























