import h5py
import csv
import sys
import os.path
import numpy as np

#check for correct number of args
if len(sys.argv) != 4:
	exit("Error: Improper usage. \nCorrect usage: csvToh5.py <input data csv> <input label csv> <output filename>")

#check if data file exists
if not os.path.isfile(sys.argv[1]):
	exit("Data CSV file path is invalid.")


#check if label file exists
if not os.path.isfile(sys.argv[2]):
	exit("Label CSV file path is invalid.")

#read the data file
with open(sys.argv[1]) as csvfile:
	reader = csv.reader(csvfile)
	dataList = list(reader)

#create an NDArray with it and force the type to float64
dataListNP = np.asarray(dataList, dtype=np.float64)

#read the label file
with open(sys.argv[2]) as csvfile:
	reader = csv.reader(csvfile)
	labelList = list(reader)

#create an NDArray with it and force the type to float64
labelListNP = np.asarray(labelList, dtype=np.float64)


#save it all to an H5 file with a data set "data" and a data set "label"
f = h5py.File(sys.argv[3], 'w')
f.create_dataset('data', data=dataListNP)

if sys.argv[2].__len__() > 0:
        f.create_dataset('label', data=labelListNP)

f.close()
