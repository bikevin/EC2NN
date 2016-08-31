import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import csv

caffe.set_mode_cpu()

if len(sys.argv) != 5 and len(sys.argv) != 6:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file> <filepath to testing data> <userdir> <list of individual points to analyze>")

if not os.path.exists(sys.argv[4]):
    exit("Error: Invalid userdir")

netFilePath = sys.argv[4] + '/' + sys.argv[1]
modelFilePath = sys.argv[4] + '/' + sys.argv[2]
testFilePath = sys.argv[4] + '/' + sys.argv[3]

if not os.path.isfile(netFilePath):
	exit("Error: File path to net prototxt is invalid.")
if not os.path.isfile(modelFilePath):
	exit("Error: File path to .caffemodel file is invalid.")
if not os.path.isfile(testFilePath):
	exit("Error: File path to testing data is invalid.")

points = []

if len(sys.argv) == 6:
	points = sys.argv[5].split(',')
	try:
		points = [int(i) for i in points]
	except ValueError:
		print 'non-int value in list of points'

data = h5py.File(testFilePath, 'r')



refNet = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)
net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)

reference = np.zeros((1, refNet.blobs['data'].data.shape[1]))
refNet.blobs['data'].data[...] = reference
#double check this starting point
#refNet.forward(start='Inner1')
refNet.forward(start='inner1')

#for layer in net.layers:
#	print layer.type
#for name in net._layer_names:
#	print name

finalValues = [0] * data.get('data').shape[1]

pointOutArray = []

for k in range(data.get('data').shape[0]):
	net.forward()
	multipliers = ()
	layerTypes = []
	multipliers += (net.blobs['data'].data - refNet.blobs['data'].data,)
	layerTypes.append('Input')
	for i in range(net.layers.__len__()):
		layerType = net.layers[i].type
		if layerType == "InnerProduct":
			multipliers += (net.layers[i].blobs[0].data,)
			layerTypes.append(layerType)
		elif layerType == "TanH" or layerType == "Sigmoid" or layerType == "ReLU":
			multipliers += (net.blobs[net._layer_names[i]].data - refNet.blobs[net._layer_names[i]].data,)
			layerTypes.append("Nonlinearity")
		else:
			continue
	
	currentValues = []
	#populate currentValues
	currentValues = np.asarray(multipliers[multipliers.__len__() - 1])
	for i in range(layerTypes.__len__() - 1):
		pos = multipliers.__len__() - i - 2
		workingValue = np.transpose(np.asarray(multipliers[pos]))
		if currentValues.shape[0] == workingValue.shape[0]:
			currentValues = np.tile(currentValues, (1, workingValue.shape[1]))
			currentValues = np.multiply(currentValues, workingValue)
		if currentValues.shape[0] == workingValue.shape[1]:
			currentValues = np.dot(workingValue, currentValues)
	for i in range(finalValues.__len__()):
		finalValues[i] += currentValues[i]
	if k + 1 in points:
		pointOutArray.append(currentValues)
	
#plt.bar(range(currentValues.__len__()), currentValues)
#plt.savefig(sys.argv[4] + '/images/importance.png')
#plt.close()

#print np.asarray(finalValues).shape
#print np.asarray(pointOutArray).shape

pointOutArray = np.asarray(pointOutArray)
pointOutArray = np.reshape(pointOutArray, (pointOutArray.shape[1], -1))

np.savetxt(sys.argv[4] + '/importance.out', finalValues, delimiter=',')
np.savetxt(sys.argv[4] + '/individualImportance.out', pointOutArray, delimiter=',')

print "Analysis Complete"
