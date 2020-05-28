# TextDetect

First, you can generate samples by youself , depending on what background you want. It means you can train a better model facing your target. Second, it's an algorithm like EAST and TextMountain.

## caffe environment

Of course, before you trainning, you should do something. For the fact that we use caffe framework, you should compile it before anything. Note that, we use caffe's python interface, so a python environment is needed. So far, python2.7 is tested. If you have a different position where python located, change it in the Makefile.config, you know it.Command line is like this :
> make all -j8<br>
> make pycaffe<br>

## train

Before training, you should get some pictures that does not have any text on it. Sample generateing algorithm will draw text on it. For the copyright reason, we only select 3 pictures located at caffe-master/examples/text_detect/images. You can add more pictures associate with your situation. Same, font files is located at caffe-master/examples/text_detect/fonts. You can even control which character you want to display on the picture, character set is located at caffe-master/examples/text_detect/char_set, but it needs some changes in the Sample generateing algorithm. If you have done things above, next you can create a lmdb database for training. It's a simple thing. Enter the caffe-master directory, run ./example/text_detect/train.py db. Time to train your model, run ./example/text_detect/train.py train. After training, model file is written to model/vgg_ht_mask/ directory.

## test 
Simple , just run ./example/text_detect/train.py test. Result is shown.

## reference
> [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2)<br/>
> [TextMountain: Accurate Scene Text Detection via Instance Segmentation](https://arxiv.org/abs/1811.12786)
