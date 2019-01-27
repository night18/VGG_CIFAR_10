'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.01.27
Description: Train VGG-like network
=======================================================================================
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras.optimizers import SGD
from pprint import pprint
import os
import numpy as np
# import util


# cifar_10_dir = "cifar-10"
models_dir = "models"
# train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)

def vgg_11():
	'''
	conv-64  -> conv-64 + maxpool  -> conv-128        -> conv-128 + maxpool -> 
	conv-256 -> conv-256 + maxpool -> conv-512        -> conv-512           ->
	FC-200   ->       FC-100       -> FC-10 + softmax
	'''
	
	model = Sequential()
	
	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	# model.add( MaxPool2D(pool_size=(3, 3), strides = 1 ) )

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	# model.add( MaxPool2D(pool_size=(3, 3), strides = 1 ) )

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	# model.add( MaxPool2D(pool_size=(3, 3), strides = 1 ) )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Flatten() )
	model.add( Dense(200, activation='relu') )

	model.add( Dense(100, activation='relu') )

	model.add( Dense(10, activation='softmax') )

	return model

def trainModel(model_name, train_data, train_labels, epochs = 5, learning_rate = 0.01):
	model = None
	storage_path = models_dir + "/" + model_name + "_" + str(learning_rate) + ".h5"

	if model_name == "vgg_11":
		model = vgg_11()

	if model != None:
		pprint("Start training model: " + model_name)
		model.compile(loss = tf.keras.losses.categorical_crossentropy,
						optimizer=SGD(lr=learning_rate),
						metrics=['accuracy'])

		hist = model.fit(train_data, train_labels, epochs = epochs, batch_size = 50 )

		tf.keras.models.save_model(
			model,
			storage_path,
			overwrite = True,
			include_optimizer=True
			)

		pprint("Successfully save the model at " + storage_path)
	return model

def loadModel(model_name, learning_rate = 0.01):
	storage_path = models_dir + "/" + model_name + "_" + str(learning_rate) + ".h5"
	try:
		pprint(storage_path)
		model = tf.keras.models.load_model(
			storage_path,
			compile = False
		)
		model.compile(loss = tf.keras.losses.categorical_crossentropy,
						optimizer=SGD(lr=learning_rate),
						metrics=['accuracy'])
	except Exception as e:
		model = None
		pprint(e)
	finally:
		return model