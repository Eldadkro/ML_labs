"""
@author: katerina
"""
from math import sqrt
import numpy as np
#import datasets 
from sklearn import datasets

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


#import datasets 
Iris = datasets.load_iris()

data = Iris.data[0:100,:]
lb = Iris.target[0:100] # true labeling

x1 = np.zeros((20,5))
x2 = np.zeros((20,5))
x1[0:20,0:4]=data[0:20,:]
x2[0:20,0:4]=data[50:70,:]
x1[0:20,4]=0
x2[0:20,4]=1
TrainingSet = np.concatenate((x1,x2),axis=0)

i=51
test_row=data[i,0:4]
for train_row in TrainingSet:
        min=100;
        dist = euclidean_distance(test_row[0:4], train_row[0:4])
        if dist<min:
           min=dist
           result=int(train_row[4])
print("The point", i, "is in the group", result)




