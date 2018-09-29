
from __future__ import print_function
import sys
from keras.datasets import mnist
from keras.layers import Input
from multiprocessing import Pool
import keras
import random
import numpy as np
from collections import defaultdict
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from _test_comb_utils_ import *
from utils import normalize, constraint_light


(_,_),(x_test,_) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

input_tensor = Input(shape = (28, 28, 1))
model1 = Model1(input_tensor = input_tensor)
model2 = Model2(input_tensor = input_tensor)
model3 = Model3(input_tensor = input_tensor)

model1_dict = defaultdict(list)
model2_dict = defaultdict(list)
model3_dict = defaultdict(list)

cover1_dict = defaultdict(bool)
cover2_dict = defaultdict(bool)
cover3_dict = defaultdict(bool)

init_coverage(model1, model1_dict, cover1_dict)
init_coverage(model2, model2_dict, cover2_dict)
init_coverage(model3, model3_dict, cover3_dict)
for i in range(10):
	model1_dict = defaultdict(list)
	model2_dict = defaultdict(list)
	model3_dict = defaultdict(list)

	cover1_dict = defaultdict(float)
	cover2_dict = defaultdict(float)
	cover3_dict = defaultdict(float)

	init_coverage(model1, model1_dict, cover1_dict)
	init_coverage(model2, model2_dict, cover2_dict)
	init_coverage(model3, model3_dict, cover3_dict)
	for a in range(10):
        	sys.stdout.flush()
        	print(a)
        	print('---------')
        	gen_img = np.expand_dims(random.choice(x_test), axis = 0)
        	update_coverage(gen_img, model1, model1_dict, cover1_dict)
        	update_coverage(gen_img, model2, model2_dict, cover2_dict)
        	update_coverage(gen_img, model3, model3_dict, cover3_dict)
        	print("Model1: %.3f, Model2: %.3f, Model3: %.3f" %(covered(cover1_dict), covered(cover2_dict), covered(cover3_dict)))
        	print("average coverage: %.3f" %((covered(cover1_dict) + covered(cover2_dict) + covered(cover3_dict))/3))
	print('-----Next Run------')
