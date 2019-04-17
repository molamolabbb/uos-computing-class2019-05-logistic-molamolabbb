#!/usr/bin/env python
# coding: utf-8

# In[109]:


import csv
import matplotlib.pyplot as plt
import math 
# Your code for regression and R2 here

def predict(alpha, beta, x_i):
    return alpha+beta*x_i

def mean(x):
    return float(sum(x))/len(x)
            
def variance(x):
    x_2 = [i**2 for i in x]
    return float(mean(x_2)-mean(x)**2)
def covariance(x,y):
    x_y = [i*j for i,j in zip(x,y)]
    return mean(x_y)-mean(x)*mean(y)

def correlation(x,y):
    return float(covariance(x,y)/(math.sqrt(variance(x))*math.sqrt(variance(y))))

def linear_regression_least_squares(x,y):
    beta = covariance(x,y)/variance(x)
    alpha = mean(y)-beta*mean(x)
    return float(alpha), float(beta)
    
def r_squared(alpha, beta, x, y):
    ss_tot = 0
    ss_res = 0
    m = mean(y)
    ss_tot = sum([(i-m)**2 for i in y])
    ss_res = sum([(y[i]-predict(alpha,beta,x[i]))**2 for i in range(len(y)) ])
    return 1-(ss_res/ss_tot)

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
    # Now, use the dataset with your regression functions to answer the exercise questions
    print("Names of the columns")
    print(header)
    print("First row of data ->variable to predict")
    print(data[0], " -> ", target[0])

    # Plot, regression here

    # Example of writing out the R2.txt file, with 0.0 guess for coefficient of correlation
    fout = open('results.txt', 'w')
    for i in range(13):
        column = [row[i] for row in data]  # get the column
        target  # target is always the median house value
        alpha, beta = linear_regression_least_squares(column,target)
        plt.clf()
        plt.scatter(column ,target)
        f_min = predict(alpha, beta, min(column))
        f_max = predict(alpha, beta, max(column))
        plt.plot([min(column),max(column)],[f_min,f_max],'r')
        plt.savefig(str(i)+".png")
        R2 = r_squared(alpha,beta,column,target)  # Fill with the real value from your code
        fout.write('%f,%f,%f\n' % (alpha, beta, R2))  # One line per variable
    fout.close()




