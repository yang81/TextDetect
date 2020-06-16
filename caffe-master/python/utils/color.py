#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import random
import math

class Color():
	def __init__(self):
		pass

	def randomColor(self):
		R = random.randint(0,255)
		G = random.randint(0,255)
		B = random.randint(0,255)
		return (R,G,B)

	def textColor(self):
		idx = random.randint(0,2)
		c1 = random.randint(1, 10)
		c2 = random.randint(1, 10)
		c3 = random.randint(1, 10)

		if idx == 0:
			#black
			return (0 + c1,0 + c2,0 + c3)
		elif idx == 1:
			#white
			return (255-c1,255-c2,255-c3)
		else:
			#random
			return self.randomColor()

	def equalColor(self, c1, c2):
		if (c1[0] == c2[0]) and (c1[1] == c2[1]) and (c1[2] == c2[2]):
			return True
		else:
			return False

	def colorDistance(self, rgb_1, rgb_2):
		R_1,G_1,B_1 = rgb_1
		R_2,G_2,B_2 = rgb_2
		rmean = (R_1 +R_2 ) / 2.
		R = R_1 - R_2
		G = G_1 -G_2
		B = B_1 - B_2
		return math.sqrt((2+rmean/256.)*(R**2)+4*(G**2)+(2+(255-rmean)/256.)*(B**2))

	def distanceRandomColor(self, color, distance):
		while True:
			tmp_color = self.randomColor()
			if self.colorDistance(tmp_color, color) > distance:
				return tmp_color


if __name__ == '__main__':
	pass