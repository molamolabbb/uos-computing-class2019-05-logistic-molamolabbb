from pytest import approx
from logistic import logistic_fn, logistic_fn_prime, logistic, logistic_prime_j, logistic_log_likelihood, logistic_log_likelihood_prime
from math import exp, log

def test_logistic():
    assert logistic_fn(0.) == approx(1./2.)
    assert logistic_fn(1.) == approx(1./(1. + exp(-1)))

    assert logistic_fn_prime(0.) == approx(1./2.*(1.-1./2.))
    
    assert logistic([0], [0, 0]) == approx(1./2.)

    assert [logistic_prime_j([0], [0, 0], 0), logistic_prime_j([0], [0, 0], 1)] == approx([1./4., 0])

    assert logistic_log_likelihood([0], 1, [0, 0]) == log(1./2)

    assert logistic_log_likelihood_prime([0], 1, [0, 0]) == [1./2, 0.]

    # write some more tests

def test_logistic_regression_sgd():
    pass
    # b = logistic_regression_sgd()
