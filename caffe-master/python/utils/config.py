#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import lmdb
import os


class Config():
	def __init__(self):
		#default image path
		self.image_path = 'examples/text_detect/images/'

		#db we use to save images when training
		self.db_path = 'examples/text_detect/train_db'

		#fonts we use to draw text on image, u can add more if u want
		self.font_root = 'examples/text_detect/fonts/'
		self.font_file = ['simhei.ttf', 'simkai.ttf', 'simsun.ttc']

		#limit font size
		self.font_size_min = 14
		self.font_size_max = 64

		#foreground text color and background color distance
		self.color_distance = 10

		#use to add gauss noise
		self.std_min = 0.005
		self.std_max = 0.02

		#character set used to draw text on image, we use part of characters in gb2312 character set  
		self.chs_labels = 'examples/text_detect/char_set/words.txt'
		self.eng_labels = 'examples/text_detect/char_set//lexicon.txt'

		#train init config
		self.solver_file = 'models/vgg_ht_mask/solver.proto'
		self.copy_weight = 'models/vgg_ht_mask/vgg_ht_mask_iter_40000.caffemodel'
		self.restore_state = 'models/vgg_ht_mask/vgg_ht_mask_iter_40000.solverstate'
		self.niter = 100000

		#test init config
		self.model = 'models/vgg_ht_mask/test.proto'
		self.weights = 'models/vgg_ht_mask/vgg_ht_mask_iter_40000.caffemodel'
		self.image = 'examples/text_detect/test_image/test.png'

	def __del__(self):
		pass


if __name__ == '__main__':
	pass