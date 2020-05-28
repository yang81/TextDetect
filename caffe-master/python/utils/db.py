#!/usr/bin/env python
# -*- coding: utf-8 -*-

__metaclss__=type

import lmdb
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.config import Config

class CreateLMDB():
	def __init__(self):
		self.config = Config()

		self.img_path = []
		self.img_path.append(self.config.image_path)
		self.map_size=1099511627776

		self.env = None
		self.txn = None
		self.count = 0

	def __commintCheck(self):
		if self.count%1000 == 0:
			self.txn.commit()
			self.txn = self.env.begin(write = True)
			print 'commit ', self.count/1000, ' batch'

	def __put(self, path):
		self.txn = self.env.begin(write = True)
		for root, dirs, files in os.walk(path):
			for f in files:
				img = Image.open(os.path.join(root, f))
							
				img_mode = img.mode
				if img_mode != "RGB":
					img = img.convert('RGB')

				min_side = min(img.height, img.width)
				if min_side < 512:
					if img.height < img.width:
						img0 = img.resize((512, 512), resample=Image.BICUBIC, box=(0,0, min_side, min_side))
						img1 = img.resize((512, 512), resample=Image.BICUBIC, box=(img.width-min_side,0, img.width, img.height))

						img_str = img0.tobytes()
						self.txn.put(key = str(self.count), value = img_str)
						self.count += 1
						self.__commintCheck()

						img_str = img1.tobytes()
						self.txn.put(key = str(self.count), value = img_str)
						self.count += 1
						self.__commintCheck()

					else:
						img0 = img.resize((512, 512), resample=Image.BICUBIC, box=(0,0, min_side, min_side))
						img1 = img.resize((512, 512), resample=Image.BICUBIC, box=(0,img.height-min_side, img.width, img.height))

						img_str = img0.tobytes()
						self.txn.put(key = str(self.count), value = img_str)
						self.count += 1
						self.__commintCheck()

						img_str = img1.tobytes()
						self.txn.put(key = str(self.count), value = img_str)
						self.count += 1
						self.__commintCheck()
				else:
					img0 = img.crop(box=(0,0, 512, 512))
					img1 = img.crop(box=(img.width-512, 0, img.width, 512))
					img2 = img.crop(box=(0, img.height-512, 512, img.height))
					img3 = img.crop(box=(img.width-512, img.height-512, img.width, img.height))
					img4 = img.crop(box=(img.width/2 - 256, img.height/2 -256, img.width/2 + 256, img.height/2 +256))

					img_str = img0.tobytes()
					self.txn.put(key = str(self.count), value = img_str)
					self.count += 1
					self.__commintCheck()

					img_str = img1.tobytes()
					self.txn.put(key = str(self.count), value = img_str)
					self.count += 1
					self.__commintCheck()

					img_str = img2.tobytes()
					self.txn.put(key = str(self.count), value = img_str)
					self.count += 1
					self.__commintCheck()

					img_str = img3.tobytes()
					self.txn.put(key = str(self.count), value = img_str)
					self.count += 1
					self.__commintCheck()

					img_str = img4.tobytes()
					self.txn.put(key = str(self.count), value = img_str)
					self.count += 1
					self.__commintCheck()

	def addImagePath(self, path):
		self.img_path.append(path)

	def write(self):
		self.env = lmdb.open(self.config.db_path, map_size=self.map_size)

		for e in self.img_path:
			self.__put(e)
			self.txn.commit()
		
		self.env.close()

		print 'done, total ', self.count, ' images'


	def read(self, i):
		env = lmdb.open(self.config.db_path)
		imgs = env.begin()

		v = imgs.get(str(i))

		env.close()

		img = Image.frombytes("RGB", (512,512), v)
		img.show()


if __name__ == '__main__':
	db = CreateLMDB()
	db.write()

	'''
	for i in range(5000, 5010):
		db.read(i)
	'''