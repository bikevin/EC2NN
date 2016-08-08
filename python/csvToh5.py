import h5py
import csv
import sys
import os.path
import numpy as np

if len(sys.argv) != 4:
	exit("Error: Improper usage. \nCorrect usage: csvToh5.py <input data csv> <input label csv> <output filename>")

if not os.path.isfile(sys.argv[1]):
	exit("Data CSV file path is invalid.")


if not os.path.isfile(sys.argv[2]):
	exit("Label CSV file path is invalid.")

with open(sys.argv[1]) as csvfile:
	reader = csv.reader(csvfile)
	dataList = list(reader)

dataListNP = np.asarray(dataList, dtype=np.float64)

with open(sys.argv[2]) as csvfile:
	reader = csv.reader(csvfile)
	labelList = list(reader)

labelListNP = np.asarray(labelList, dtype=np.float64)



f = h5py.File(sys.argv[3], 'w')
f.create_dataset('data', data=dataListNP)
f.create_dataset('label', data=labelListNP)
f.close()


		
	
