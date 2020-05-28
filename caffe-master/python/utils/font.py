#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import random
from PIL import ImageFont

class FontObj():
	def __init__(self, config, font_size = 0):
		self.root = config.font_root
		self.files = config.font_file

		if font_size != 0:
			self.size = font_size
		else:
			self.size = random.randint(config.font_size_min, config.font_size_max)

		self.style = random.randint(0,2)
		self.style = 1
		self.font = ImageFont.truetype(self.root + self.files[self.style], self.size)


if __name__ == '__main__':
	pass