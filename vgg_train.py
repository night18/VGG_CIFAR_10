'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.01.27
Description: Train VGG-like network
=======================================================================================
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import SGD

import os
import numpy as np
import pickle

# import util


# cifar_10_dir = "cifar-10"
models_dir = "models"
history_dir = "history"
# train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)

def vgg_11():
	'''
	conv-64  -> conv-64 + maxpool  -> conv-128        -> conv-128 + maxpool -> 
	conv-256 -> conv-256 + maxpool -> conv-512        -> conv-512           ->
	FC-200   ->       FC-100       -> FC-10 + softmax
	'''
	
	model = Sequential()
	
	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same', input_shape=(32,32,3) ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())

	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())

	model.add( Flatten() )
	model.add( Dense(200, activation='relu') )

	model.add( Dense(100, activation='relu') )

	model.add( Dense(10, activation='softmax') )

	return model

def vgg_11_02():
	'''
	conv-64  -> conv-64 + maxpool  -> conv-128        -> conv-128 + maxpool -> 
	conv-512        -> conv-512    ->
	FC-200   ->       FC-100       -> FC-10 + softmax
	'''
	
	model = Sequential()
	
	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same', input_shape=(32,32,3) ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )

	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )
	# model.add(Dropout(0.4))
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )

	model.add( Conv2D(128, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )
	# model.add(Dropout(0.4))
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )
	model.add(BatchNormalization())


	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Flatten() )
	model.add( Dense(200, activation='relu') )

	model.add( Dense(100, activation='relu') )

	model.add( Dense(10, activation='softmax') )

	return model

def vgg_11_03():
	'''
	conv-64  -> conv-64 + maxpool  ->
	conv-256 -> conv-256 + maxpool -> conv-512        -> conv-512           ->
	FC-200   ->       FC-100       -> FC-10 + softmax
	'''
	
	model = Sequential()
	
	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same', input_shape=(32,32,3) ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )

	model.add( Conv2D(64, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )
	# model.add(Dropout(0.4))
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )

	model.add( Conv2D(256, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )
	# model.add(Dropout(0.4))
	# model.add( MaxPool2D(pool_size=(2,2),strides=(2,2)) )
	model.add( MaxPool2D(pool_size=(3, 3), strides = (1,1) ) )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add(BatchNormalization())
	model.add( Activation('relu') )

	model.add( Conv2D(512, kernel_size = (3,3), padding = 'same' ) )
	model.add( Activation('relu') )

	model.add( Flatten() )
	model.add( Dense(200, activation='relu') )

	model.add( Dense(100, activation='relu') )

	model.add( Dense(10, activation='softmax') )

	return model


def trainModel(model_name, train_data, train_labels, validation_data, validation_labels, epochs = 5, learning_rate = 0.01):
	model = None
	h5_storage_path = models_dir + "/" + model_name + "_" + str(learning_rate) + ".h5"
	hist_storage_path = history_dir + "/" + model_name + "_" + str(learning_rate)
	
	if model_name == "vgg_11":
		model = vgg_11()
	elif model_name == "vgg_11_02":
		model = vgg_11_02()
	elif model_name == "vgg_11_03":
		model = vgg_11_03()

	if model != None:
		print("Start training model: " + model_name)
		model.compile(loss = tf.keras.losses.categorical_crossentropy,
						optimizer = SGD(lr = learning_rate),
						metrics = ['accuracy'])

		hist = model.fit(
				train_data, 
				train_labels, 
				epochs = epochs,
				batch_size = 13,
				validation_data=(validation_data, validation_labels)
			)
		
		#Save model and weight
		save_model(
			model,
			h5_storage_path,
			# '/home/chunwei/Documents/class/CPE691/HW1/model.h5',
			overwrite=True,
			include_optimizer=True
		)

		#Save the history of training
		with open(hist_storage_path, 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)

		print("Successfully save the model at " + h5_storage_path)
	return model

def loadModel(model_name, learning_rate = 0.01):
	h5_storage_path = models_dir + "/" + model_name + "_" + str(learning_rate) + ".h5"
	
	try:
		model = load_model(
		    h5_storage_path,
		    # '/home/chunwei/Documents/class/CPE691/HW1/model.h5',
		    custom_objects=None,
		    compile=True
		)

	except Exception as e:
		model = None
		print(e)
	finally:
		return model