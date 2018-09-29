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
from _test_combin_utils_ import *
from utils import normalize, constraint_light
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('seeds', help="number of seeds of input", type=int)
args = parser.parse_args()


random.seed(1001)


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
nb_adv = 0
#init_coverage(model1, model1_dict, cover1_dict)
#init_coverage(model2, model2_dict, cover2_dict)
#init_coverage(model3, model3_dict, cover3_dict)

#model_dicts = [model1_dict, model2_dict, model3_dict]
#cover_dicts = [cover1_dict, cover2_dict, cover3_dict]
#models = [model1, model2, model3]

#model_iterable = []
#for i in range(len(model_dicts)):
#	model_iterable.append((models[i], model_dicts[i], cover_dicts[i])) 

#print(model_iterable)
# with Pool(processes=3) as p:
#	p.starmap(init_coverage, model_iterable)
init_coverage(model1, model1_dict, cover1_dict)
init_coverage(model2, model2_dict, cover2_dict)
init_coverage(model3, model3_dict, cover3_dict)

for a in range(1,args.seeds):
	sys.stdout.flush()
	print('Test Input:%d' %(a))
	test_case = np.expand_dims(random.choice(x_test), axis = 0)
	label1, label2, label3 = np.argmax(model1.predict(test_case)[0]), np.argmax(model2.predict(test_case)[0]), np.argmax(model3.predict(test_case)[0])
	if not label1 == label2 == label3:
	#	update_iterables = []
	#	for i in len(model_iterable):
	#		update_iterables.append((gen_img, model_iterable[i]))
	#	with Pool(processes = 3) as p:
	#		p.starmap(update_coverage, update_iterables)
		update_coverage(test_case, model1, model1_dict, cover1_dict)
		update_coverage(test_case, model2, model2_dict, cover2_dict)
		update_coverage(test_case, model3, model3_dict, cover3_dict)
		print("Model1: %.3f, Model2: %.3f, Model3: %.3f" %(covered(cover1_dict), covered(cover2_dict), covered(cover3_dict)))
		print("average coverage: %.3f" %((covered(cover1_dict) + covered(cover2_dict) + covered(cover3_dict))/3))
		nb_adv += 1
	orig_label = label1
#	with Pool(processes = 3) as p:
#		loss_cov1, loss_cov2, loss_cov3 = p.starmap(loss_coverage, model_iterable)
	loss_cov1 = loss_coverage(model1, model1_dict, cover1_dict)
	loss_cov2 = loss_coverage(model2, model2_dict, cover2_dict)
	loss_cov3 = loss_coverage(model3, model3_dict, cover3_dict)
	loss1 = - K.mean(model1.get_layer('before_softmax').output[...,orig_label])
	loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
	loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
	optim_function = K.mean((loss1 + loss2 + loss3) + 0.1*(loss_cov1 + loss_cov2 + loss_cov3))
	grad_ip = normalize(K.gradients(optim_function, input_tensor)[0])
	iterate = K.function([input_tensor], [loss1, loss2, loss3, loss_cov1, loss_cov2, loss_cov3, grad_ip])
	for _ in range(10):
		loss_val1, loss_val2, loss_val3, loss_cov1, loss_cov2, loss_cov3, gradients = iterate([test_case])
		grads_value = constraint_light(gradients)
		test_case += grads_value * 10
		pred1,pred2,pred3 = np.argmax(model1.predict(test_case)[0]),np.argmax(model2.predict(test_case)[0]),np.argmax(model3.predict(test_case)[0])
		if not pred1 == pred2 == pred3:
			update_coverage(test_case, model1, model1_dict, cover1_dict)
			update_coverage(test_case, model2, model2_dict, cover2_dict)
			update_coverage(test_case, model3, model3_dict, cover3_dict)
			print("Model1: %.3f, Model2: %.3f, Model3: %.3f" %(covered(cover1_dict), covered(cover2_dict), covered(cover3_dict)))
			print("average coverage: %.3f" %((covered(cover1_dict) + covered(cover2_dict) + covered(cover3_dict))/3))
			nb_adv += 1
			break	
	if (a)%2000 == 0:
		print('Number of Test Inputs: %d' %(a))
		print('Number of Corner Case Inputs found: %d' %(nb_adv))
	
