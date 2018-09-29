from __future__ import print_function
from collections import defaultdict
import numpy as np
import keras
from keras.models import Model
from multiprocessing import Pool
from keras import backend as K
from itertools import combinations, product	
import random
import tensorflow as tf
random.seed(1001)

def init_coverage(model, model_dict, coverage_dict):
#def init_coverage(model_iterable):
#	model = model_iterable[0]
#	model_dict = model_iterable[1]
#	coverage_dict = model_iterable[2]
	layers = [layer.name for layer in model.layers if 'input' not in layer.name and 'flatten' not in layer.name]
	for layer in layers:
		for num_neuron in range(model.get_layer(layer).output_shape[-1]):
			model_dict[(layer, num_neuron)] = ['False']
	for k in range(1, len(layers)):
		layer1 = model.get_layer(layers[k-1])
		layer2 = model.get_layer(layers[k])
		for n1 in range(layer1.output_shape[-1]-1):
			for n2 in range(n1+1, layer1.output_shape[-1]):
				for n3 in range(layer2.output_shape[-1]):
					coverage_dict[(layer1.name, layer2.name, n1, n2, n3)] = False

def update_coverage(test_input, model, model_dict, coverage_dict):
#def update_coverage(update_iterable):
#	test_input = update_iterable[0]
#	model = update_iterable[1]
#	model_dict = update_iterable[2]
#	coverage_dict = update_iterable[3]
	activations = defaultdict(list)
	layers = [layer.name for layer in model.layers if 'input' not in layer.name and 'flatten' not in layer.name]
	for layer in layers:
		layer_model = Model(inputs = model.input, outputs = model.get_layer(layer).output)
		layer_op = layer_model.predict(test_input)
		scaled = scale(layer_op[0])
		for num_neuron in range(scaled.shape[-1]):
			if np.mean(scaled[..., num_neuron]) > 0:
				activations[(layer, num_neuron)] = ['True']
			else:
				activations[(layer, num_neuron)] = ['False']
		
	for key, value in activations.items():
		model_dict[key].extend(value)
	
	parameters = []
	for k in range(len(layers)):
		layer1 = model.get_layer(layers[k-1])
		layer2 = model.get_layer(layers[k])
		for n1 in range(layer1.output_shape[-1]):
			for n2 in range(n1+1, layer1.output_shape[-1]):
				for n3 in range(layer2.output_shape[-1]):
					coverage_dict[(layer1.name, layer2.name, n1, n2, n3)] = update_dict(model_dict, layer1.name, layer2.name, n1, n2, n3)

def update_dict(model_dict, layer1, layer2, n1, n2, n3):
	parameter_values = [model_dict[(layer1, n1)], model_dict[(layer1, n2)], model_dict[(layer2, n3)]]
	combinations_ = list(combinations(range(3), 2))
	cov = []
	for row in combinations_:
		coverage = pair_coverage(parameter_values[:][row[0]], parameter_values[:][row[1]])
		if coverage < 1:
			return False
			break
	return True

def pair_coverage(col_1, col_2):
	arr = []
	for i in range(len(col_1)):
		row = [col_1[i], col_2[i]]
		arr.append(row)
	return len(np.unique(arr, axis=0))/4
	
def covered(coverage_dict):
	covered = len([v for v in coverage_dict.values() if v])
	return covered/len(coverage_dict) 


def scale(X, rmax=1, rmin=0):
	X_std = (X - X.min())/(X.max()-X.min())
	X_scaled = (rmax - rmin) * X_std + rmin
	return X_scaled

def loss_coverage(model, model_dict, cover_dict):
	not_covered = [(layer1, layer2, n1, n2, n3) for (layer1, layer2, n1, n2, n3), v in cover_dict.items() if not v]
	layer1, layer2, n1, n2, n3 = random.choice(not_covered)
	parameter_values = [model_dict[(layer1, n1)], model_dict[(layer1, n2)], model_dict[(layer2, n3)]]
	all_combinations = list(product([['True'], ['False']], [['True'],['False']], [['True'],['False']]))
	for i in range(len(all_combinations)):
		if all_combinations[i][:] not in parameter_values:
			to_cover = all_combinations[i][:]
			break
	loss_coverage_coeff = []
	for i in range(len(to_cover)):
		if to_cover[i] == 'True':
			loss_coverage_coeff.append(1)
		else:
			loss_coverage_coeff.append(-1)
	loss_coverage_coeff = np.asarray(loss_coverage_coeff)
	loss_coverage = loss_coverage_coeff.item(0) * K.mean(model.get_layer(layer1).output[...,n1]) + loss_coverage_coeff.item(1) * K.mean(model.get_layer(layer1).output[...,n2]) + loss_coverage_coeff.item(2) * K.mean(model.get_layer(layer2).output[..., n3])
	return loss_coverage 

