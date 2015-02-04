import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pylab import *
import csv

def load(fileName):
	y = []
	X = []

	reader = csv.reader(open(fileName))
	for row in reader:
		y.append(int(float(row[-1])))
		X.append([float(x) for x in row[:-1]])

	X = np.array(X)
	y = np.array(y)
	return X,y


X,y = load('datasets/trainr3.csv')
Xtest1,ytest1 =  load('datasets/testr30.csv')
Xtest2,ytest2 =  load('datasets/testr31.csv')


score1 = []
score2 = []
kMax = 10

for k  in range(1,kMax):
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(X,y)
	score1.append(neigh.score(Xtest1,ytest1))
	score2.append(neigh.score(Xtest2,ytest2))


#plot scores
xlabel('k')
ylabel('accuracy')
title('k-NN accuracy vs k')
grid(True)
plot(np.arange(1,kMax),score1,label = 'testset2')
plot(np.arange(1,kMax),score2,label = 'testset1')
legend(loc='upper left')
savefig('q2.png')
print('q2 done')