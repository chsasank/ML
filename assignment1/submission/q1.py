import numpy as np
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from pylab import *
import csv

def load(fileName):
	y = []
	X =[]
	reader = csv.reader(open(fileName))

	for row in reader:
		y.append(row[0])
		X.append([float(x) for x in row[1:]])

	X = np.array(X)
	y = np.array(y)
	return X,y


X,y = load('datasets/image_recognition.csv')
N = y.size
kMax = 10

####q1b : 5 fold cross validation
scoreK = []
for k  in range(1,kMax):
	neigh = KNeighborsClassifier(n_neighbors=k)
	scores = cross_validation.cross_val_score(neigh,X,y,cv = 5)
	scoreK.append(np.mean(scores))

print(scoreK)
plot(np.arange(1,kMax),scoreK)
xlabel('k')
ylabel('accuracy')
title('k-NN 5-fold cross validation accuracy vs k')
grid(True)
savefig('q1b.png')
print('q1b done')



#### q1a : leave one out accuracy

#take a random n size sample from the dataset
n = 10000
sampleIndices = np.random.randint(N,size = n)
X = X[sampleIndices,:]
y = y[sampleIndices]
loo = cross_validation.LeaveOneOut(n)

scoreK = []
for k  in range(1,kMax):
	neigh = KNeighborsClassifier(n_neighbors=k)
	scores = cross_validation.cross_val_score(neigh,X,y,cv = loo)
	scoreK.append(np.mean(scores))


print(scoreK)
plot(np.arange(1,kMax),scoreK)
xlabel('k')
ylabel('accuracy')
title('k-NN leave one out accuracy vs k')
grid(True)
savefig('q1a.png')
print('q1a done')
