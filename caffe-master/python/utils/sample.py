#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import random
import math
import collections
from PIL import Image, ImageDraw, ImageFont
from utils.font import FontObj
from utils.color import Color
import skimage.util
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time


class WordList():
	def __init__(self, config):
		self.gb2312set = config.chs_labels
		self.english = config.eng_labels

		with open(self.gb2312set, 'r') as f:
			self.chs = f.readlines()
	
		for i in range(len(self.chs)):
			self.chs[i] = self.chs[i].replace('\n', '')

		'''
		with open(self.english, 'r') as f:
			self.eng = f.readlines()
	
		for i in range(len(self.eng)):
			self.eng[i] = self.eng[i].replace('\n', '')
		'''

	def get(self, index):
		char = ''

		if index <= 0 :
			pass
		else:
			char = self.chs[index]

		return char

	def get_array(self, arr):
		char = ''
		for e in arr:
			if e <= 0 :
				continue
			char += self.chs[e]

		return char
		

class Generator():
	def __init__(self, word_list):
		self.words = word_list
		self.pos = 1
		self.pre_pos = []
		self.dic = []

	def getRandomTxt(self, chs, count):
		word_list = []
		if chs == True:
			word_list = self.words.chs
		else:
			word_list = self.words.eng

		self.rand_count = count
		self.rand_txt = ''
		self.rand_labels = random.sample(range(1, len(word_list)), int(count))

		for e in self.rand_labels:
			if chs == True:
				self.rand_txt += word_list[e]
			else:
				self.rand_txt += word_list[e]
				self.rand_txt += ' '

		if chs == True:
			return self.rand_txt
		else:
			self.rand_txt = self.rand_txt[0:len(self.rand_txt)-1]
			return self.rand_txt

	def getRandomEnglishTxt(self, count):
		word_list = []

		word_list = self.words.chs

		eng_list = []
		txt_slice = []

		for i in range(count):
			#get character in GB2312 sector 3 randomly
			c = random.randint(166, 259)
			eng_list.append(c)
			txt_slice.append(word_list[c])

		self.dic.append(collections.OrderedDict(zip(eng_list, txt_slice)))

		return  txt_slice




	def traverseWordList(self, chs, count):
		word_list = []
		txt_slice = []
		k = []

		if chs == True:
			word_list = self.words.chs
		else:
			word_list = self.words.eng

		self.traverse_txt = ''
		list_len = len(word_list)


		start = self.pos
		end = start + count

		if start < list_len and end <= list_len:
			#normal
			txt_slice = word_list[start:end]
			self.traverse_labels = range(start, end)
			k = range(start, end)
		elif start >= list_len:
			#case 1
			start = start % list_len + 1
			end = end % list_len + 1
			txt_slice = word_list[start:end]
			self.traverse_labels = range(start, end)
			k = range(start, end)
		elif start < list_len and end > list_len:
			#case 2
			end = end % list_len + 1
			txt_slice = word_list[start:list_len] + word_list[1:end]
			a = range(start, list_len)
			b = range(1, end)
			self.traverse_labels = a + b
			k = a + b
		else:
			print 'unknow case'

		self.pre_pos.append(start)
		self.pos = end
		
		for e in txt_slice:
			if chs == True:
				self.traverse_txt += e
			else:
				self.traverse_txt += e
				self.traverse_txt += ' '

		v = txt_slice

		self.dic.append(collections.OrderedDict(zip(k, v)))

		self.traverse_count = len(txt_slice)

		if chs == True:
			return self.traverse_txt
		else:
			self.traverse_txt = self.traverse_txt[0:len(self.traverse_txt)-1]
			return self.traverse_txt

	def getPrePos(self):
		if len(self.pre_pos) != 0:
			self.pos = self.pre_pos[0]

		self.pre_pos = []
		self.dic = []

	def clearPrePos(self):
		self.pre_pos = []


	def getDict(self):
		return self.dic

	def clearDict(self):
		self.dic = []


class RecogniseSample():
	def __init__(self, config, generator):
		self.config = config
		self.generator = generator
		self.color = Color()

	def getSample(self):
		#get text from label set
		text = ""
		label = []
		blank_count = random.randint(0,5)
		
		if random.choice([True, False]):
			text = self.generator.traverseWordList(True, self.config.recognise_label_len - blank_count)
			self.generator.clearDict()
			self.generator.clearPrePos()
			label = self.generator.traverse_labels
		else:
			text = self.generator.getRandomTxt(True, self.config.recognise_label_len - blank_count)
			label = self.generator.rand_labels

		for i in range(blank_count):
			l_pos = random.randint(0, len(label) - 1)
			t_pos = random.randint(0, len(label) - 1)

			label.insert(l_pos, 0)
			t = text.decode("utf-8")
			t = t[0:t_pos] + ' ' +  t[t_pos:]
			text = t.encode("utf-8")

		#get font
		font = FontObj(self.config)

		t_w, t_h = font.font.getsize(text.decode('utf-8'))
		text_color = self.color.textColor()
		back_color = self.color.distanceRandomColor(text_color, self.config.color_distance)

		img = Image.new("RGB", (t_w, t_h), back_color)

		draw = ImageDraw.Draw(img)
		draw.text((0, 0), text.decode('utf-8'), font = font.font, fill=text_color)

		alpha = random.uniform(0, math.pi/16.)
		sign = math.pow(-1, random.randint(1,2))

		w = int(t_w + t_h*math.tan(alpha))
		h = t_h
		t = t_h*math.tan(alpha) if sign < 0 else 0

		M = np.array([[1, sign*math.tan(alpha), t],
					[0, 1, 0]], np.float32)

		img = np.asarray(img)

		img = cv2.warpAffine(img,M,(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=back_color)

		img = cv2.resize(img, (self.config.recognise_img_width, self.config.recognise_img_height), interpolation=cv2.INTER_LINEAR)

		kernel = np.ones((2,2), np.uint8)

		if random.choice([True, False]):
			img = cv2.erode(img, kernel, 1)
		
		if random.choice([True, False]):
			img = cv2.dilate(img, kernel, 1)

		if random.choice([True, False]):
			img = cv2.GaussianBlur(img, (3,3), 1)

		if self.config.recognise_gray:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#im_ = Image.fromarray(img)
		#im_.show()
		#im_.save(self.config.image_path + "img_" +  str(random.randint(0,100)) + ".png")

		img = img.astype('float32')
		img /= 255

		if self.config.recognise_gray:
			return (img,np.array(label, dtype=np.float32))
		else:
			return (img.transpose((2,0,1)),np.array(label, dtype=np.float32))



class ImageSample():
	def __init__(self, config, generator, img):
		self.config = config
		self.generator = generator
		#self.count = count
		self.pad = 1.
		self.next_y_pos = float(random.randint(1, 256))
		self.next_x_pos = self.pad
		#line space
		self.line_space = 1.

		#text pad
		self.text_pad = 1
		self.image = None
		self.label_info = []

		self.block_theta = 0.
		self.block_angle = 0.
		self.block_patch = None
		self.block_txt_color = []
		self.block_crop = None
		self.block_back_color = []
		self.block_fonts = []
		self.block_heights = []

		self.color = Color()


		self.patch_info = {'patch_w_half':0., 'patch_off_x':0., 'patch_off_y':0.}

		if img != None :
			self.image = img.copy()
		
	def __genetrateAImage(self):
		self.image_type = 1 if self.image == None else 2
		
		if self.image_type == 1:
			self.block_back_color = self.color.randomColor()
			self.image = Image.new("RGB", (512, 512), self.block_back_color)

		return self.image

	def __getColor(self):
			idx = random.randint(0,2)
			c1 = random.randint(1, 10)
			c2 = random.randint(1, 10)
			c3 = random.randint(1, 10)

			if idx == 0:
				return (0 + c1,0 + c2,0 + c3)
			elif idx == 1:
				return (255-c1,255-c2,255-c3)
			else:
				return self.color.randomColor()

	def __equalColor(self, c1, c2):
		if (c1[0] == c2[0]) and (c1[1] == c2[1]) and (c1[2] == c2[2]):
			return True
		else:
			return False

	def __ColourDistance(self, rgb_1, rgb_2):
		   R_1,G_1,B_1 = rgb_1
		   R_2,G_2,B_2 = rgb_2
		   rmean = (R_1 +R_2 ) / 2.
		   R = R_1 - R_2
		   G = G_1 -G_2
		   B = B_1 - B_2
		   return math.sqrt((2+rmean/256.)*(R**2)+4*(G**2)+(2+(255-rmean)/256.)*(B**2))

	def __get_std(self, size):
			if size < 32 :
				return self.config.std_min
			else:
				std = self.config.std_min + (self.config.std_max - self.config.std_min)/(self.config.font_size_max - self.config.font_size_min) * (size - self.config.font_size_min)
				return std

	def __getChordLengthX(self, y):
		#default circle center is (512,512) , radius is (512./2.)
		r = 512./2.
		difference = math.pow(r, 2) - math.pow(y-r, 2)
		if difference <0:
			print y
			print difference
		chordLength = math.pow(math.pow(r, 2) - math.pow(y-r, 2), 0.5)
		return  (r - chordLength, 2 * chordLength)


	def __getRealFontSize(self, font, item):
		k = item[0]
		v = item[1]

		w = 0.
		h = 0.
		offset = 0.

		#python 3
		#cwh = font.font.getsize(v)
		(tmp_w, tmp_h) = font.font.getsize(v.decode('utf-8'))
		if k == 1 or k == 2:
			w = 0.5*tmp_w
			h = tmp_h
			offset = 0.
		elif k == 4 or k == 5 or k == 6 or k == 7 or k == 34 or k == 49 or k == 50 or k == 64 or k == 65 or k == 91 or k == 92 or k == 154:
			w = 0.5*tmp_w
			h = tmp_h
			offset = (float(tmp_w) - w)/2.
		elif k == 8 or k == 11 or k == 39 or k == 40 or k == 43 or k == 72 or k == 73 or k == 75:
			w = 0.7*tmp_w
			h = tmp_h
			offset = (float(tmp_w) - w)/2.
		elif k >= 286 and k <= 322:
			w = 0.7*tmp_w
			h = tmp_h
			offset = (float(tmp_w) - w)/2.
		elif k >= 13 and k <= 30 and k%2 == 1:
			w = 0.5*tmp_w
			h = tmp_h
			offset = 0.5*tmp_w
		elif k >= 13 and k <= 30 and k%2 == 0:
			w = 0.5*tmp_w
			h = tmp_h
			offset = 0.
		elif k == 66 or k == 67 or k == 68:
			w = 0.5*tmp_w
			h = tmp_h
			offset = 0.
		else:
			w = float(tmp_w)
			h = float(tmp_h)
			offset = 0.

		return (w, h, offset)

	def __getPatchInfo(self, (block_w, block_h)):
		half_w = block_w/2.
		half_h = block_h/2.

		patch_w_half = math.sqrt(half_w*half_w + half_h*half_h)

		patch_off_x = patch_w_half - half_w
		patch_off_y = patch_w_half - half_h
		
		self.patch_info = {'patch_w_half':patch_w_half, 'patch_off_x':patch_off_x, 'patch_off_y':patch_off_y}

		return (patch_w_half,patch_off_x,patch_off_y)
		
	def __getTxtBlockWithPos(self, chs, traverse, flag):
		
		count = 0
		g = self.generator
		txt = ''

		n = random.randint(1, 77)
		txt_y = self.next_y_pos + n * self.line_space


		if (txt_y) >= self.image.height:
			return (False, False,0, 0, 0, 0, False)
		
		row_column = random.choice([True, False])

		lines = random.randint(1, 5) if row_column else 2


		block_w = 0.
		block_h = 0.

		for i in range(lines):
			font = FontObj(self.config)
			max_word_count = int(512/font.size)

			if flag > 0:
				count = flag
			elif flag == 0:
				count = random.randint(1, int(max_word_count))
			else:
				count = max_word_count

			if traverse == True:
				if chs == True:
					txt = g.traverseWordList(chs, count)
				else:
					#english donnt need to traverse, just random select count characters
					txt = g.getRandomEnglishTxt(count)
				
			else:
				if chs == True:
					txt = g.getRandomTxt(chs, count)
				else:
					txt = g.getRandomEnglishTxt(count)
				

			d = g.getDict()
			tmp_w = 0.
			tmp_h = 0.

			for k, v in d[i].items():
				(w, h, offset) = self.__getRealFontSize(font, (k,v))
				tmp_w += w
				if h > tmp_h:
					tmp_h = h

		
			self.block_fonts.append(font)
			self.block_heights.append(tmp_h)


			if row_column:
				if tmp_w > block_w:
					block_w = tmp_w

				block_h += tmp_h
				block_h += self.line_space
			else:
				if tmp_h > block_h:
					block_h = tmp_h

				space_w = font.font.getsize(' '.decode('utf-8'))[0]
				block_w += tmp_w + space_w


		#get patch info
		(patch_w_half,patch_off_x,patch_off_y) = self.__getPatchInfo((block_w, block_h))

		n = random.randint(int(patch_w_half), int(patch_w_half) + 10)
		range_left = n * self.pad + self.next_x_pos

		txt_x = range_left
	
		if (txt_y + block_h + self.pad + 2 * patch_off_y) > self.image.height:
			if traverse == True:
				g.getPrePos()
				self.block_fonts = []
				self.block_heights = []

			return (False, False,0, 0, 0, 0, False)

		if (txt_x + block_w + self.pad + 2 * patch_off_x) > self.image.width :
			if traverse == True:
				g.getPrePos()
				self.block_fonts = []
				self.block_heights = []

			return (False, False,0, 0, 0, 0, False)

		if traverse == True:
			return (True,row_column,txt_x,txt_y,block_w,block_h, True)
		else:
			return (True,row_column,txt_x,txt_y,block_w,block_h, True)

	
	def __addGraphicNoise(self):
		#graphic noise
		draw_geometry = ImageDraw.Draw(self.image)

		#line noise have bad influence on some character or word (Chinese word '一')
		'''
		for line_count in range(random.randint(3,8)):
			line_color = self.color.randomColor()
			line_width = random.randint(1, 5)
			x1 = random.randint(0, self.image.width)
			y1 = random.randint(0, self.image.height)
			x2 = random.randint(0, self.image.width)
			y2 = random.randint(0, self.image.height)

			draw_geometry.line((x1, y1, x2, y2) ,fill = line_color, width=line_width)
		'''

		x1 = random.randint(0, self.image.width)
		y1 = random.randint(0, self.image.height)
		x2 = random.randint(0, self.image.width)
		y2 = random.randint(0, self.image.height)

		g_color = self.color.randomColor()
		g_fill_color = self.color.randomColor()
		g_width = random.randint(1, 5)
		b_fill = random.randint(0,1)
		g_shape = random.randint(0,2)
		if g_shape == 0:
			if b_fill == 0:
				draw_geometry.rectangle([x1,y1, x2, y2], fill=g_fill_color, outline = g_color, width=g_width)
			else:
				draw_geometry.rectangle([x1,y1, x2, y2], outline = g_color, width=g_width)
		elif g_shape == 1:
			if b_fill == 0:
				draw_geometry.ellipse([x1,y1, x2, y2], fill=g_fill_color, outline = g_color, width=g_width)
			else:
				draw_geometry.ellipse([x1,y1, x2, y2], outline = g_color, width=g_width)
		else:
			vertex_count = random.randint(3,5)
			vertex_list = []
			for e in range(vertex_count):
				vx = random.randint(0, self.image.width)
				vy = random.randint(0, self.image.height)
				vertex_list.append((vx, vy))

			if b_fill == 0:
				draw_geometry.polygon(vertex_list, fill=g_fill_color, outline = g_color)
			else:
				draw_geometry.polygon(vertex_list, outline = g_color)

	def __addGaussNoise(self):
		#gauss noise
		self.block_patch = np.asarray(self.block_patch)
		self.block_patch = skimage.util.img_as_float(self.block_patch)

		min_font = 1000
		for e in self.block_fonts:
			if e.size < min_font:
				min_font = e.size

		std = self.__get_std(min_font)
		noise = np.random.randn(self.block_patch.shape[0], self.block_patch.shape[1], self.block_patch.shape[2]) * std
		self.block_patch += noise
		self.block_patch = np.where(self.block_patch >=1, 0.99999, self.block_patch)
		self.block_patch = skimage.util.img_as_ubyte(self.block_patch)
		self.block_patch = Image.fromarray(self.block_patch)

	def __getRotatedCoordinate(self, cx, cy, x, y, alpha):
		radius = math.sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy))

		cos_a = (x - cx)/(radius + 0.00001)
		sin_a = (cy - y)/(radius + 0.00001)

		rx = cx + radius * (cos_a*math.cos(alpha) - sin_a*math.sin(alpha))
		ry = cy - radius * (sin_a*math.cos(alpha) + cos_a*math.sin(alpha))

		return (rx, ry)

	def __generateLabels(self, (x,y,patch_w_half, theta), cxy_w_h):
		#compute rotated text center
		c_x = float(x) + float(patch_w_half)
		c_y = float(y) + float(patch_w_half)
		alpha = float(theta)
		for e in cxy_w_h:
			(rx, ry) = self.__getRotatedCoordinate(c_x, c_y, e[0], e[1], alpha)

			#draw_geometry = ImageDraw.Draw(self.image)
			#draw_geometry.point([rx,ry], fill=(255,0,0))

			label_line = []

			label_line.append(rx)
			label_line.append(ry)
			label_line.append(e[2])
			label_line.append(e[3])
			label_line.append(alpha)

			'''
			(rx, ry) = self.__getRotatedCoordinate(c_x, c_y, e[4], e[5], alpha)
			draw_geometry.point([rx,ry], fill=(255,0,0))
			label_line.append(rx)
			label_line.append(ry)

			(rx, ry) = self.__getRotatedCoordinate(c_x, c_y, e[6], e[7], alpha)
			draw_geometry.point([rx,ry], fill=(255,0,0))
			label_line.append(rx)
			label_line.append(ry)
			'''

			self.label_info.append(label_line)

	def __rotateBlock(self):
		self.block_theta = float(math.pow(-1, random.randint(1,2)) * random.uniform(0, math.pi/2.))
		#self.block_theta = float((math.pi/16.)*np.random.randn())
		self.block_angle = float(self.block_theta*180./math.pi)

		#rotate block
		self.block_patch = self.block_patch.rotate(self.block_angle, resample=Image.BICUBIC, fillcolor=(0,0,0,0))

	def __getBlockTxtColor(self, (x,y,patch_w_half)):
		#compute backcolor in the image, select a new text color with the given color distance
		self.block_crop = self.image.crop(box=(int(round(x)),int(round(y)), int(round(x)) +  int(round(patch_w_half * 2)), int(round(y)) + int(round(patch_w_half * 2)) ))
		crop_np = np.asarray(self.block_crop)
		self.block_back_color = (crop_np[:,:,0].mean(), crop_np[:,:,1].mean(), crop_np[:,:,2].mean())

		while True:
			self.block_txt_color = self.__getColor()
			if self.__ColourDistance(self.block_back_color, self.block_txt_color) > self.config.color_distance:
				break

	def __pasteBlockToImage(self, (x,y)):
		self.block_crop = self.block_crop.convert('RGBA')
		self.block_crop.alpha_composite(self.block_patch)
		self.block_crop = self.block_crop.convert('RGB')
		self.image.paste(self.block_crop, box=(int(round(x)),int(round(y))))

	def __clearBlock(self):
		self.block_fonts = []
		self.block_heights = []
		self.generator.clearDict()
		self.generator.clearPrePos()

	def __Txt(self, chs, traverse, flag, multi_instance=False):
		self.__genetrateAImage()

		get_pos = True
		c = 0

		patch_off_x = 0.
		patch_off_y = 0.

		previous_h = 0.

		if self.image_type == 1:
			#self.__addGraphicNoise()
			pass
		
		while get_pos == True:
			(get_pos,row_column,x,y,block_w,block_h, line_prob) = self.__getTxtBlockWithPos(chs, traverse, flag)
			if get_pos == True:

				patch_w_half = self.patch_info['patch_w_half']
				patch_off_x = self.patch_info['patch_off_x']
				patch_off_y = self.patch_info['patch_off_y']

				self.next_x_pos = x + block_w + 2 * patch_off_x

				line_max_h = y + block_h + 2 * patch_off_y
				if line_max_h > previous_h:
					previous_h =  line_max_h

				#compute backcolor in the image, select a new text color with the given color distance
				self.__getBlockTxtColor((x,y,patch_w_half))
				

				#create a transparent image to draw text on it
				self.block_patch = Image.new("RGBA", (int(round(patch_w_half * 2)), int(round(patch_w_half * 2))), color=(0,0,0,0))
				draw_patch = ImageDraw.Draw(self.block_patch)

				draw_start_x = patch_off_x
				draw_start_y = patch_off_y

				cxy_w_h = []

				#draw text on patch
				if traverse == False:
					#draw_patch.text((draw_start_x, draw_start_y), txt.decode('utf-8'), font = font.font, fill=self.block_txt_color)
					pass
				else:
					d = self.generator.getDict()
					for e, f, d_i in zip(d, self.block_fonts, range(len(d))):
						cy_offset = 0.
						if row_column:
							draw_start_x = patch_off_x
						else:
							draw_start_y = patch_off_y

							if ( (d_i == 0) and (self.block_heights[0] < self.block_heights[1])):
								cy_offset = random.randint(0, int(self.block_heights[1]/2.))
								draw_start_y += cy_offset

							if ( (d_i == 1) and (self.block_heights[0] > self.block_heights[1])):
								cy_offset = random.randint(0, int(self.block_heights[0]/2.))
								draw_start_y += cy_offset

						tmp_h = 0.
						tmp_w = 0.
						for (k, v), e_i in zip(e.items(), range(len(e))):
							(cw, ch, coffset) = self.__getRealFontSize(f, (k,v))
							if e_i == 0:
								draw_start_x -= coffset

							draw_patch.text((draw_start_x, draw_start_y), v.decode('utf-8'), font = f.font, fill=self.block_txt_color)

							draw_start_x += cw
							tmp_w += cw
							if ch > tmp_h:
								tmp_h = ch

						if row_column:
							draw_start_y += self.line_space
							draw_start_y += tmp_h

							line_w = tmp_w
							line_h = tmp_h

							line_cx = patch_off_x + line_w/2. + x
							line_cy = draw_start_y - self.line_space - line_h/2. + y

							left_cx = patch_off_x + x
							left_cy = draw_start_y - self.line_space - line_h/2. + y

							right_cx = patch_off_x + line_w + x
							right_cy = draw_start_y - self.line_space - line_h/2. + y

							#cxy_w_h.append([line_cx, line_cy, line_w, line_h])
							cxy_w_h.append([line_cx, line_cy, line_w, line_h, left_cx, left_cy, right_cx, right_cy])
						else:
							draw_start_x += f.font.getsize(' '.decode('utf-8'))[0]

							line_w = tmp_w
							line_h = tmp_h

							line_cx = draw_start_x - f.font.getsize(' '.decode('utf-8'))[0] - line_w/2. + x
							line_cy = draw_start_y + line_h/2. + y

							left_cx = draw_start_x - f.font.getsize(' '.decode('utf-8'))[0] - line_w + x
							left_cy = draw_start_y + line_h/2. + y

							right_cx = draw_start_x - f.font.getsize(' '.decode('utf-8'))[0] + x
							right_cy = draw_start_y + line_h/2. + y

							#cxy_w_h.append([line_cx, line_cy, line_w, line_h])
							cxy_w_h.append([line_cx, line_cy, line_w, line_h, left_cx, left_cy, right_cx, right_cy])

				#rotate text block
				self.__rotateBlock()
				
				#gauss noise
				self.__addGaussNoise()

				#paste block back to image
				self.__pasteBlockToImage((x,y))

				#generate labels
				self.__generateLabels((x,y,patch_w_half, self.block_theta), cxy_w_h)

				#clear block 
				self.__clearBlock()

				c += 1
				
				if flag == 1 and c == 1 and multi_instance == False:
					break

			else:
				if c == 0 :
					get_pos = True
					
				if multi_instance == True and line_prob == True:
					n = random.randint(1, 77)
					self.next_y_pos += previous_h + n * self.line_space
					self.next_x_pos = self.pad
					get_pos = True

		#self.image.show()
		img_array = np.asarray(self.image, dtype='f')
		#img_array -= 127.5
		img_array /= 255

		return (img_array.transpose((2,0,1)),self.label_info)


	def randomLenTxt(self, chs):
		return self.__Txt(chs, False, 0, True)

	def traverseRandomLenTxt(self, chs):
		return self.__Txt(chs, True, 0)
	
	def randomSingleWordTxt(self, chs, multi_instance=False):
		return self.__Txt(chs, False, 1, multi_instance)

	def traverseSingleTxt(self, chs, multi_instance=False):
		return self.__Txt(chs, True, 1, multi_instance)

	def randomMaxWordTxt(self, chs):
		return self.__Txt(chs, False, -1)

if __name__ == '__main__':
	pass