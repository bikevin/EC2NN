import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import caffe
import h5py
import copy
import sys

caffe.set_mode_cpu()

if len(sys.argv) != 3:
	exit("Error: Incorrect number of arguments. \nUsage: net_analyzer.py <file path to model prototxt> <file path to .caffemodel file>")

#generate the net
net = caffe.Net(str(sys.argv[1]), str(sys.argv[2]), caffe.TEST)

#TODO NEXT:
#create list for variable importance
#pull that size from the size of the input layer
#create list for all the individual variable graphs

#figure out some way to handle models of any size dynamically
#at least the math's been worked out

