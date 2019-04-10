from pytest import approx
from logistic import logistic_fn, logistic_fn_prime, logistic
from math import exp

def test_logistic():
    assert logistic_fn(0.) == approx(1./2.)
    assert logistic_fn(1.) == approx(1./(1. + exp(-1)))

    assert logistic_fn_prime(0.) == approx(1./2.*(1.-1./2.))
    
    assert logistic([0], [0, 0]) == approx(1./2.)

    # write some more tests

def test_logistic_regression_sgd():
    pass
    # b = logistic_regression_sgd()
