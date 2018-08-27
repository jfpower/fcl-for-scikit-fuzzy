# -*- coding: utf-8 -*-
"""

@author: james.power@mu.ie Created on Wed Aug  1 14:14:46 2018
Based on Annex A if the IEEE 1855-2016 standard
"""

# Each hedge takes a membership function and modifies it,
# returning a 'hedged' membership function of the same size.

# Two of these (above/below) need the x-values also.

import numpy as np
import skfuzzy.membership as skmemb
import extramf


def above(x, mf):
    ''' A.1: above(mf)=0 if x<x_max, 1-mf if x>= x_max; cf below'''
    x_max = x.max()
    new_mf = 1 - mf
    new_mf[np.nonzero(x < x_max)] = 0
    return new_mf


def any_of(mf):
    '''A.2: any(mf) = 1.. ('any' is a Python built-in)'''
    return np.ones_like(mf)


def below(x, mf):
    '''A.3: below(mf) = 0 if x>x_max, 1-mf(x) if x<x_max; cf above'''
    x_max = x.max()
    new_mf = 1 - mf
    new_mf[np.nonzero(x > x_max)] = 0
    return new_mf


def extremely(mf):
    '''A.4: extremely(mf) = mf ** 3'''
    return mf ** 3


def intensify(mf):
    '''A.5: 2*mf^2 if mf<0.5, 1-2*(1-mf)^2 otherwise; cf seldom '''
    new_mf = np.zeros_like(mf)
    under, over = np.nonzero(mf <= 0.5), np.nonzero(mf > 0.5)
    new_mf[under] = 2 * (mf[under] ** 2)
    new_mf[over] = 1 - (2 * ((1 - mf[over]) ** 2))
    return new_mf


def more_or_less(mf):
    '''A.6: more_or_less(mf) = mf ^ 3'''
    return mf ** (1/3)


def norm(mf):
    '''A.7:  norm divides by maximum'''
    return mf / mf.max()


def is_not(mf):
    '''A.8: not(mf) = 1-mf ('not' is a Python keyword)'''
    return 1 - mf


def plus(mf):
    '''A.9: more_or_less(mf) = mf ^ (5/4)'''
    return mf ** (5/4)


def seldom(mf):
    '''
        A.10: (mf/2)^(1/2) if mf<=0.5, and 1-((1-mf)/2)^(1/2) otherwise;
        cf intensify
    '''
    new_mf = np.zeros_like(mf)
    under, over = np.nonzero(mf <= 0.5), np.nonzero(mf > 0.5)
    new_mf[under] = np.sqrt(mf[under] / 2)
    new_mf[over] = 1 - np.sqrt((1 - mf[over]) / 2)
    return new_mf


def slightly(mf):
    '''A.11: Defined as: intensify [ norm (plus S AND not very S) ]'''
    def min_and(x, y):
        ''' Let's implement AND as the elementwise min of two arrays'''
        return np.minimum.reduce([x, y])
    return intensify(norm(min_and(plus(mf), is_not(very(mf)))))


def somewhat(mf):
    '''A.12: somewhat(mf) = mf ^ (1/2)'''
    return mf ** (1/2)


def very(mf):
    '''A.13: very(mf) = mf ^ 2'''
    return mf ** 2


# List of all hedges, maps name to function and 'takes-x' flag
all_hedges = {
    'above':        (above, True),
    'any':          (any_of, False),
    'below':        (below, True),
    'extremely':    (extremely, False),
    'intensify':    (intensify, False),
    'more_or_less': (more_or_less, False),
    'norm':         (norm, False),
    'not':          (is_not, False),
    'plus':         (plus, False),
    'seldom':       (seldom, False),
    'slightly':     (slightly, False),
    'somewhat':     (somewhat, False),
    'very':         (very, False),
}


def test_all_hedges(x, y):
    results = []
    hedgenames = sorted(all_hedges.keys())  # Want them in alphabetical order
    for name in hedgenames:
        func, takes_x = all_hedges[name]
        if takes_x:
            results.append(func(x, y))
        else:
            results.append(func(y))
    return (hedgenames, results)


if __name__ == '__main__':
    x = np.arange(0, 100)
    y = skmemb.gaussmf(x, 50, 20)
    (titles, data) = test_all_hedges(x, y)
    extramf.visualise_all(x, [y]+data, ['original (gaussian)']+titles)

