import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np
import util
import vgg_train
import matplotlib.pyplot as plt

cifar_10_dir = "cifar-10"
epochs = 40
validation_number = 5000
train_number = 50000 - validation_number

def testModel(model, model_name, x_test, y_test):
	score = model.evaluate(x_test, y_test)
	print("=============================================")
	print("Test perforemance of model " + model_name)
	print('Test loss:'+ str(score[0]))
	print('Test accuracy:'+ str(score[1]))


def plot_performance(histories, name_list, isloss = True, isVal = False):
	#isloss means whether plot loss. If True, plot loss, nor plot accuracy

	perforemance = 'loss' if isloss else 'acc'
	val = 'val_' if isVal else ''

	perforemance = val + perforemance
	# print(perforemance)

	for hist in histories:
		plt.plot(hist[perforemance])
	plt.xticks(np.arange(0, epochs +1 , epochs/5 ))
	plt.ylabel(perforemance)
	plt.xlabel( "epochs" )
	plt.legend( name_list , loc=0)
	plt.show()
	plt.savefig( perforemance + ".png" )



if __name__ == '__main__':
	
	#load data set 
	train_data, _, train_labels, test_data, _, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)


	train_data, validation_data = train_data[ 0:train_number ], train_data[ train_number:50000 ]
	train_labels, validation_labels = train_labels[ 0:train_number ], train_labels[ train_number:50000 ]
	# validation_data, train_data = train_data[ 0:validation_number ], train_data[ validation_number:50000 ]
	# validation_labels, train_labels = train_labels[ 0:validation_number ], train_labels[ validation_number:50000 ]
	# test_data =  test_data[0:1000]
	# test_labels = test_labels[0:1000]


	train_data, validation_data, test_data = train_data/255.0, validation_data/255.0 , test_data/255.0
	train_labels= to_categorical(train_labels,num_classes=10)
	validation_labels = to_categorical(validation_labels, num_classes=10)
	test_labels= to_categorical(test_labels,num_classes=10)

	learning_rate_list = [0.01, 0.0009, 0.0001]
	model_list = ["vgg_11", "vgg_11_02", "vgg_11_03"]
	histories = []
	name_list = []

	# for x in model_list:
	# 	model = vgg_train.loadModel(x)
	# 	if model == None: 
	# 		model = vgg_train.trainModel(x, train_data, train_labels, validation_data, validation_labels, epochs = epochs)
	# 	testModel(model, x, test_data, test_labels )
	# 	path = "history/{}_0.01".format(x)
	# 	histories.append( util.unpickle(path) )

	# plot_performance(histories, model_list, isloss = True, isVal=True)
	for y in model_list:
		for x in learning_rate_list:
			model = vgg_train.loadModel(y, x)

			if model == None:
				model = vgg_train.trainModel(y, train_data, train_labels, validation_data, validation_labels, epochs = epochs, learning_rate = x)

			testModel(model, y + "_"+str(x), test_data, test_labels )

			path = "history/{}_{}".format(y,x)
			name = "{}_{}".format(y,x)
			histories.append( util.unpickle(path) )
			name_list.append( name )

	plot_performance(histories, name_list, isloss = False, isVal=False)
