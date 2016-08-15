import sys
sys.path.append('/home/ubuntu/caffe-master/python')
import caffe
import os.path
import rsquared as r2
import numpy as np
from pylab import *

caffe.set_mode_cpu()

if len(sys.argv) != 5:
        exit("Error: Incorrect number of arguments. \nUsage: net_predictor.py <file path to model prototxt> <filepath to caffemodel file> <filepath to testing data> <userdir>")
        
if not os.path.isfile(sys.argv[4]):
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

net = caffe.Net(str(netFilePath), str(modelFilePath), caffe.TEST)
net.forward()

#make sure the predict network doesn't have a loss layer, but there's no real check for that

np.savetxt(sys.argv[4] + '/predict.out', net.blobs[net._blob_names[-1]].data, delimiter=',')


