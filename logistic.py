#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python

from math import exp, log
from multiple import stochastic_minimize
# You will need to import stochastic_minimize

# Functions go here
def dot(x,beta):
    return sum(x_i*beta_i for x_i, beta_i in zip(x,beta))

def logistic_fn(x):
    return 1/(1+exp(-x))

def logistic_fn_prime(x):
    return logistic_fn(x)*(1-logistic_fn(x))

def logistic(x,beta):
    return logistic_fn(dot(x,beta))

def logistic_prime_j(x, beta, j):
    if j >= 1:
        return x[j-1]*logistic(x,beta)*(1-logistic(x,beta))
    elif j == 0:
        return logistic(x,beta)*(1-logistic(x,beta))

def logistic_log_likelihood(x_i, y_i, beta):
    if y_i == 1:
        if logistic(x_i,beta)==0: return log(1E-100)
        return log(logistic(x_i,beta))
    elif y_i == 0:
        if logistic(x_i,beta)==1: return log(1E-100)
        return log(1-logistic(x_i,beta))

def logistic_log_likelihood_prime(x_i, y_i, beta):
    if y_i==0: return [logistic_prime_j(x_i,beta,j) / (1-logistic(x_i,beta)) for j in range(len(beta))]
    elif y_i==1: return [logistic_prime_j(x_i,beta,j)/logistic(x_i,beta) for j in range(len(beta))]

def negative_log_likelihood(x_i, y_i, beta):
	return -1*logistic_log_likelihood(x_i,y_i,beta)	
	

def logistic_regression_sgd(x0, x1, beta0):
    x = x0+x1
    y0 = [0 for i in range(len(x0))]
    y1 = [1 for i in range(len(x1))]
    y = y0+y1
    return stochastic_minimize(negative_log_likelihood,logistic_log_likelihood_prime,x,y,beta0,1e-6)
		
def accuracy(data, target, f, beta):
    result=0
    result0 = 0
    result1 = 0
    for d, t in zip(data,target):
        if (f(d,beta) > 0.5) and t==1: 
            result1 +=1
        elif (f(d,beta) < 0.5) and t==0:
            result0 +=1
    result = result0 + result1
    return float(result)/float(len(data))

if __name__ == "__main__":
    import csv
    diabetes = csv.reader(open('diabetes.csv'))
    header = diabetes.__next__()  # change to diabetes.next() for python2!
    data_ = list(d for d in diabetes)
    data = [[float(di) for di in d[:-1]] for d in data_]
    target = [int(d[-1]) for d in data_]
    print("Header:", header)
    # Sort data into not-diabetes (0) and diabetes (1)
    x0 = [d for d, t in zip(data, target) if t == 0]
    x1 = [d for d, t in zip(data, target) if t == 1]
    # Make some tests of your logistic regression, then try with the diabetes dataset
    beta0 = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    beta = logistic_regression_sgd(x0,x1,beta0)
    fout = open('results.txt','w')
    fout.write('accuracy : %f \n' %(accuracy(data,target,logistic,beta)))
    for b in beta:
        fout.write('%f \n' %(b))
    fout.close()
