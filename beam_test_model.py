#Script context use	: This script uses Raymotime data (https://www.lasse.ufpa.br/raymobtime/) in the context of the UFPA - ITU Artificial Intelligence/Machine Learning in 5G Challenge (http://ai5gchallenge.ufpa.br/).
#Author       		: Ailton Oliveira, Aldebaro Klautau, Arthur Nascimento, Diego Gomes, Jamelly Ferreira, Walter Frazão
#Email          	: ml5gphy@gmail.com                                          
#License		: This script is distributed under "Public Domain" license.
###################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Trains a deep NN for choosing top-K beams
Adapted by AK: Aug 7, 2018
See
https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
and
https://stackoverflow.com/questions/45642077/do-i-need-to-use-one-hot-encoding-if-my-output-variable-is-binary
See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
import csv
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate,Average
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta,Adam
from sklearn.model_selection import train_test_split
from ModelHandler import ModelHandler
from DenseMoE import DenseMoE
import numpy as np
import argparse


###############################################################################
# Support functions
###############################################################################

#For description about top-k, including the explanation on how they treat ties (which can be misleading
#if your classifier is outputting a lot of ties (e.g. all 0's will lead to high top-k)
#https://www.tensorflow.org/api_docs/python/tf/nn/in_top_k
def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape
        
        for i in range(0,y_shape[0]):            
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        
        return y

def getBeamOutput(output_file):
    
    thresholdBelowMax = 6
    
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']
        
         
    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)

    #beam_output = np.load('beam_output.npy')
    #np.save('beam_output', y)
    #np.savetxt('data.csv', y,delimiter=',')
    
    return y,num_classes


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('data_folder', help='Location of the data directory', type=str)
#TODO: limit the number of input to 3
parser.add_argument('--input', nargs='*', default=['coord'], 
choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('-p','--plots', 
help='Use this parametter if you want to see the accuracy and loss plots',
action='store_true')
args = parser.parse_args()

###############################################################################
# Data configuration
###############################################################################
tf.device('/device:GPU:0')
data_dir = args.data_folder+'/'
tgtRec = 3

if 'coord' in args.input: 
    ###############################################################################
    # Coordinate configuration
    #train
    coord_train_input_file = data_dir+'coord_input/coord_train.npz'
    coord_train_cache_file = np.load(coord_train_input_file)
    X_coord_train = coord_train_cache_file['coordinates']
    #validation
    coord_validation_input_file = data_dir+'coord_input/coord_validation.npz'
    coord_validation_cache_file = np.load(coord_validation_input_file)
    X_coord_validation = coord_validation_cache_file['coordinates']
    # print(X_coord_train)
    coord_train_input_shape = X_coord_train.shape
    # print(coord_train_input_shape)
	
    #test_data
    coord_test_input_file = data_dir+'baseline_data/coord_input/coord_test.npz'
    coord_test_cache_file = np.load(coord_test_input_file)
    X_coord_test = coord_test_cache_file['coordinates']
    print("Coord test shape is: ",X_coord_test.shape)	

if 'img' in args.input:
    ###############################################################################
    # Image configuration
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    imgDim = (360,640) # Image dimensions
    method = 1
    #train
    img_train_input_file = data_dir+'image_input/img_input_train_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_train_input_file)
    img_train_cache_file = np.load(img_train_input_file)
    X_img_train = img_train_cache_file['inputs']
    #validation
    img_validation_input_file = data_dir+'image_input/img_input_validation_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_validation_input_file)
    img_validation_cache_file = np.load(img_validation_input_file)
    X_img_validation = img_validation_cache_file['inputs']

    img_train_input_shape = X_img_train.shape
	
    img_test_input_file = data_dir+'baseline_data/image_input/img_input_test_'+str(resizeFac)+'.npz'
    img_test_cache_file = np.load(img_test_input_file)
    X_img_test = img_test_cache_file['inputs']
    img_test_input_shape = X_img_test.shape
    print("image test shape is: ",X_img_test.shape)



if 'lidar' in args.input:
    ###############################################################################
    # LIDAR configuration
    #train
    lidar_train_input_file = data_dir+'lidar_input/lidar_train.npz'
    print("Reading dataset... ",lidar_train_input_file)
    lidar_train_cache_file = np.load(lidar_train_input_file)
    X_lidar_train = lidar_train_cache_file['input']
    #np.save('lidar_train', X_lidar_train)
    #validation
    lidar_validation_input_file = data_dir+'lidar_input/lidar_validation.npz'
    print("Reading dataset... ",lidar_validation_input_file)
    lidar_validation_cache_file = np.load(lidar_validation_input_file)
    X_lidar_validation = lidar_validation_cache_file['input']

    #test_data
    lidar_test_input_file = data_dir+'baseline_data/lidar_input/lidar_test.npz'
    lidar_test_cache_file = np.load(lidar_test_input_file)
    X_lidar_test = lidar_test_cache_file['input']
    print("shape of lidar test data is: ",X_lidar_test.shape)

    #np.save('lidar_validation', X_lidar_validation)
    # print(X_lidar_train[0])
    lidar_train_input_shape = X_lidar_train.shape
    # print(lidar_train_input_shape)
	
###############################################################################
# Output configuration
#train
output_train_file = data_dir+'beam_output/beams_output_train.npz'
y_train,num_classes = getBeamOutput(output_train_file)

output_validation_file = data_dir+'beam_output/beams_output_validation.npz'
y_validation, _ = getBeamOutput(output_validation_file)

#test
output_test_file = data_dir+'baseline_data/beam_output/beams_output_test.npz'
y_test,_ = getBeamOutput(output_test_file)

##############################################################################
# Model configuration
##############################################################################

#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)

num_epochs = 10
batch_size = 32
validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam()

model = Model.load_model('my_saved_model.h5')

if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        print("Predicting the model on test data......")
        y_test=model.predict([X_lidar_test,X_coord_test])
    elif 'coord' in args.input and 'img' in args.input:
        y_test=model.predict([X_coord_test,X_img_test])    
    else:
        y_test=model.predict([X_lidar_test,X_img_test])
		
elif multimodal == 3:
    print("Predicting the model on test data......")
    y_test=model.predict([X_lidar_test,X_img_test,X_coord_test])

else:
    if 'coord' in args.input:
       y_test=model.predict(X_coord_test)
    elif 'img' in args.input:
       y_test=model.predict(X_img_test)
    else:
        y_test=model.predict(X_lidar_test)


print('predictions Shape is : ',y_test.shape)
print('Max value is: ',max(y_test[0]))
pred = np.argmax(y_test, axis = 1)[:5]
label = np.argmax(y_test, axis = 1)[:5]
print('pred is: ',pred)
print('test output  is: ',label)
print('predictions are: ',y_test)
###############################################################################
# Save the result for evaluation
###############################################################################
np.savetxt('beam_test_pred.csv', y_test, delimiter=',')
