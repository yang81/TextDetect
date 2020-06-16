#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import random
from PIL import ImageFont

true_type_fonts = []

class FontObj():
	def __init__(self, config, font_size = 0):

		if len(true_type_fonts) == 0:
			root = config.font_root
			files = config.font_file
			font_type0 = []
			font_type1 = []
			font_type2 = []

			i = config.font_size_min
			while i <= config.font_size_max:
				font0 = ImageFont.truetype(root + files[0], i)
				font1 = ImageFont.truetype(root + files[1], i)
				font2 = ImageFont.truetype(root + files[2], i)

				font_type0.append(font0)
				font_type1.append(font1)
				font_type2.append(font2)

				i += 1

			true_type_fonts.append(font_type0)
			true_type_fonts.append(font_type1)
			true_type_fonts.append(font_type2)

			self.__get(config, font_size)
		else:
			self.__get(config, font_size)

	
	def __get(self, config, font_size = 0):
		if font_size != 0:
			self.size = font_size
		else:
			self.size = random.randint(config.font_size_min, config.font_size_max)

		self.style = random.randint(0,2)
		self.font = true_type_fonts[self.style][self.size-config.font_size_min]


if __name__ == '__main__':
	pass