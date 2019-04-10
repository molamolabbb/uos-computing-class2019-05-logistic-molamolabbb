#!/usr/bin/env python

from math import exp, log
# You will need to import stochastic_minimize

# Functions go here

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
