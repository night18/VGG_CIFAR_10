import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from pprint import pprint

import numpy as np
import util
import vgg_train
import matplotlib.pyplot as plt

cifar_10_dir = "cifar-10"


def testModel(model, model_name, x_test, y_test):
	score = model.evaluate(x_test, y_test)
	pprint("=============================================\n")
	pprint("Test perforemance of model " + model_name)
	pprint('Test loss:'+ score[0])
	pprint('Test accuracy:'+ score[1])


if __name__ == '__main__':
	
	#load data set 
	train_data, _, train_labels, test_data, _, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)

	test_data =  test_data[0:1000]
	test_labels = test_labels[0:1000]

	vgg_11_model = vgg_train.loadModel("vgg_11")
	
	if vgg_11_model == None:
		vgg_11_model = vgg_train.trainModel("vgg_11", train_data, train_labels, epochs = 1)

	testModel(vgg_11_model, "vgg_11", test_data, test_labels )