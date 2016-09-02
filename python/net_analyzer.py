import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import csv

#use the CPU instead of the GPU
caffe.set_mode_cpu()

#check for the correct number of arguments
if len(sys.argv) != 5 and len(sys.argv) != 6:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file> <filepath to testing data> <userdir> <list of individual points to analyze>")

#check if userdir exists
if not os.path.exists(sys.argv[4]):
    exit("Error: Invalid userdir")

#append userdir to filepaths
netFilePath = sys.argv[4] + '/' + sys.argv[1]
modelFilePath = sys.argv[4] + '/' + sys.argv[2]
testFilePath = sys.argv[4] + '/' + sys.argv[3]

#check if these files exist
if not os.path.isfile(netFilePath):
	exit("Error: File path to net prototxt is invalid.")
if not os.path.isfile(modelFilePath):
	exit("Error: File path to .caffemodel file is invalid.")
if not os.path.isfile(testFilePath):
	exit("Error: File path to testing data is invalid.")

#convert the string of points to analyze into an array
points = []
if len(sys.argv) == 6:
	points = sys.argv[5].split(',')
	try:
		points = [int(i) for i in points]
	except ValueError:
		print 'non-int value in list of points'

data = h5py.File(testFilePath, 'r')

#initialize both the net that takes the reference measurement and the net that takes the actual measurements
#TODO probably save the reference measurement weights and call that good tbh instead of initializing a whole new net
refNet = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)
net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)

#current reference - a vector of all zeros 
#TODO find a way for users to choose their own reference inputs
reference = np.zeros((1, refNet.blobs['data'].data.shape[1]))
refNet.blobs['data'].data[...] = reference
refNet.forward(start='inner1')

#initialize the output as an array of all zeros
finalValues = [0] * data.get('data').shape[1]

#initialize the individual point output as an empty array
pointOutArray = []

#this next part is based off of the DeepLIFT paper - https://arxiv.org/pdf/1605.01713v2.pdf
#you're not going to understand anything if you don't read the paper
for k in range(data.get('data').shape[0]):
	
	#compute everything for the data point
	net.forward()
	
	#initialize the multipliers tuple
	multipliers = ()
	
	#input multiplier - the delta between the data and the reference data
	multipliers += (net.blobs['data'].data - refNet.blobs['data'].data,)
	
	#initialize the multipliers for the rest of the layesr
	for i in range(net.layers.__len__()):
		
		#check the layer types
		layerType = net.layers[i].type
		
		#for inner product (affine) layers, the multipliers are the weights 
		if layerType == "InnerProduct":
			multipliers += (net.layers[i].blobs[0].data,)
		#for non-linear layers, the multipliers are just the delta data I think
		elif layerType == "TanH" or layerType == "Sigmoid" or layerType == "ReLU":
			multipliers += (net.blobs[net._layer_names[i]].data - refNet.blobs[net._layer_names[i]].data,)
		else:
			continue
	
	#yeah the following is going to be a pretty bad explanation but whatever
	
	#we move backwards in the tuple, starting with the output
	currentValues = np.asarray(multipliers[multipliers.__len__() - 1])
	for i in range(multipliers.__len__() - 1):
		
		#counting up so we use the pos variable which counts down
		pos = multipliers.__len__() - i - 2
		
		#transpose the next multiplier - this is an artifact of the way caffe stores the data
		workingValue = np.transpose(np.asarray(multipliers[pos]))
		
		#if the next layer is a nonlinear layer
		if currentValues.shape[0] == workingValue.shape[0]:
			
			#replicate the data so it's the same size 
			currentValues = np.tile(currentValues, (1, workingValue.shape[1]))
			
			#element-wise multiply everything
			currentValues = np.multiply(currentValues, workingValue)
			
		#if it's a inner product (affine) layer
		if currentValues.shape[0] == workingValue.shape[1]:
			
			#we can use summation to delta to prove that multiplying into a affine layer is matrix multiplication
			currentValues = np.dot(workingValue, currentValues)
			
	#add the delta-inputs
	for i in range(finalValues.__len__()):
		finalValues[i] += currentValues[i]
		
	#append the point-wise importances to an array
	if k + 1 in points:
		pointOutArray.append(currentValues)
	
#point-wise array needs to be flattened to print correctly
pointOutArray = np.asarray(pointOutArray)
pointOutArray = np.reshape(pointOutArray, (pointOutArray.shape[1], -1))

#print it all out
np.savetxt(sys.argv[4] + '/importance.out', finalValues, delimiter=',')
np.savetxt(sys.argv[4] + '/individualImportance.out', pointOutArray, delimiter=',')

print "Analysis Complete"
