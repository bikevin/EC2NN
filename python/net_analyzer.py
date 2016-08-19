import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

caffe.set_mode_cpu()

if len(sys.argv) != 5:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file> <filepath to testing data> <userdir>")

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

data = h5py.File(testFilePath, 'r')



refNet = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)
net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)

reference = np.zeros((1, refNet.blobs['data'].data.shape[1]))
refNet.blobs['data'].data[...] = reference
#double check this starting point
#refNet.forward(start='Inner1')
refNet.forward(start='inner2')

#for layer in net.layers:
#	print layer.type
#for name in net._layer_names:
#	print name

finalValues = [1] * data.get('data').shape[1]
for k in range(data.get('data').shape[0]):
	net.forward()
	multipliers = ()
	layerTypes = []
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
	for value in multipliers[multipliers.__len__() - 1]:
		currentValues = value
	for i in range(layerTypes.__len__()):
		#have to go backwards in multiplier tuple
		pos = multipliers.__len__() - i - 1
		newVals = []
		for j in range(multipliers[pos].shape[1]):
			if layerTypes[pos] == "InnerProduct":
				temp = 0
				for l in range(multipliers[pos].shape[0]):
					temp += multipliers[pos][l][j] * currentValues[l]
				newVals.append(temp)
			else:
				newVals.append(currentValues[j] * multipliers[pos][0][j])
		currentValues = newVals

	deltaInput = net.blobs['data'].data - refNet.blobs['data'].data
	for i in range(deltaInput.shape[1]):
		currentValues[i] *= deltaInput[0][i]
		finalValues[i] += currentValues[i]
	
#plt.bar(range(currentValues.__len__()), currentValues)
#plt.savefig(sys.argv[4] + '/images/importance.png')
#plt.close()

np.savetxt(sys.argv[4] + '/importance.out', currentValues, delimiter=',')

print "Analysis Complete"
