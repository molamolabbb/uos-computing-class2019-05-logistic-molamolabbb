#!/usr/bin/env python3

import csv
from math import erf, sqrt
from randomness import randnr
import numpy as np
import decimal
from regression import linear_regression_least_squares
rand = randnr(3)
def in_random_order(x):
	#"Returns an iterator that presents the list x in a random order"
	indices = [i for i, _ in enumerate(x)]
    # "inside-out" Fisher-Yates shuffle. Step through the list, and at
    # each point, exchange the current element with a random element
    # in the list (including itself)
	for i in range(len(indices)):
		j = (rand.randint() // 65536) % (i+1)  # The lower bits of our random generator are correlated!
		indices[i], indices[j] = indices[j], indices[i]
	for i in indices:
	 	yield x[i]

# Functions go here!
def predict(x_i, beta):
	return beta[0] + sum(beta_i*x for beta_i, x in zip(beta[1:],x_i))
def loss(x_i, y_i, beta):
	#loss_ls=[beta[i+1]*x_i[i] for i in range(len(x_i))]
	#loss_ls.append(beta[0])
	#loss_ls.append(-y_i)
	#return (sum(loss_ls))**2
	return (predict(x_i, beta)- y_i)**2

def dloss(x_i, y_i, beta):
	#loss_ls=[beta[i+1]*x_i[i] for i in range(len(x_i))]
	#loss_ls.append(beta[0])
	#loss_ls.append(-y_i)
	res = predict(x_i,beta)-y_i
	dloss_ls = [res*2*x for x in [1]+x_i]
	return dloss_ls

def stochastic_minimize(f, df, x, y, theta0, alpha0=0.001, iterations=50):
	min_theta = None
	min_value = float('inf')
	alpha = alpha0
	theta = theta0
	iterations_without_improvement = 0
	data = list(zip(x,y))

	while iterations_without_improvement < iterations:
		value = sum([f(x_i,y_i,theta) for x_i, y_i in data])
		if value < min_value:
			iterations_without_improvement = 0
			min_theta, min_value = theta, value
			alpha = alpha0
		else:
			iterations_without_improvement += 1
			alpha *= 0.9
			theta = min_theta
		for x_i, y_i in in_random_order(data):
			gradient_i = df(x_i,y_i,theta)
			theta = [theta[i]-alpha*gradient_i[i] for i in range(len(theta))]
	return min_theta

def r_squared(x,y,beta):
	m = (1./len(y))*sum(y)
	ss_res = sum([loss(xi,yi,beta) for xi,yi in zip(x,y)])
	ss_tot = sum((y_i-m)**2 for y_i in y)
	R_sqr = 1- float(ss_res)/float(ss_tot)
	print (ss_res)
	print (ss_tot)
	return R_sqr

if __name__ == "__main__":
# Here, we load the boston dataset
	boston = csv.reader(open('boston.csv'))  # The boston housing dataset in csv format
    # First line contains the header, short info for each variable
	header = boston.next()  # In python2, you might need boston.next() instead
    # Data will hold the 13 data variables, target is what we are trying to predict
	data, target = [], []
	for row in boston:
        # All but the last are the data points
		data.append([float(r) for r in row[:-1]])
        # The last is the median house value we are trying to predict
		target.append(float(row[-1]))
    # Now, use the dataset with your regression functions to answer the exercise question
	print("Names of the columns")
	print(header)
	print("First row of data ->variable to predict")
	print(data[0], " -> ", target[0])

    # The alpha parameter must be tuned low so that we don't jump too far
	start = [0,-0.412775,0.142140,-0.648490,6.346157,-33.916055,9.102109,-0.123163,1.091613,-0.403095,-0.025568,-2.157175,0.033593,-0.950049]
	'''
	start = [0]
	for i in range(13):
		column = [row[i] for row in data]
		alpha, beta = linear_regression_least_squares(column,target)
		start.append(beta)
	'''
	# take the starting parameters as 0 for beta0 then the intercepts from the individual fits
	output = stochastic_minimize(loss, dloss, data, target, start,1e-6)
	
	R2 = r_squared(data,target,output)
	#print R2
	# Also need to calculate the full R^2!
  # Example of writing out the results.txt file
	fout = open('results.txt', 'w')
	for param in output:
		fout.write('%f\n' % (param)) # One line per variabl
	fout.write('R2: {0}'.format(R2))
	fout.close()

