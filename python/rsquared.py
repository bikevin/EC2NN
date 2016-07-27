import csv
import numpy as np

def getR2FromFile():
	actual = []
	with open("actual.txt") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			line[0] = float(line[0])
			actual += line

	predicted = []
	with open("predicted.txt") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			line[0] = float(line[0])
			predicted += line
	return calculateR2(actual, predicted)

	

def calculateR2(pred, act):
	actualMean = np.mean(act)

	sumSquares = 0
	for value in act:
		sumSquares += np.square(value - actualMean)

	sumResid = 0
	for i in range(0, len(act)):
		sumResid += np.square(act[i] - pred[i])

	return 1 - sumResid/sumSquares
