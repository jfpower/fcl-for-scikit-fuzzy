# -*- coding: utf-8 -*-
"""
    Hedge functions, based on the definitions in the IEEE standard.
    Each function here maps a mf to a new mf.
    I'm following Annex A in IEEE 1855-2016, definintions A.1-A.13.
    @author: james.power@mu.ie Created on Wed Aug  1 14:14:46 2018
"""

# Each hedge takes a membership function and modifies it,
# returning a 'hedged' membership function of the same size.


import numpy as np
import skfuzzy.membership as skmemb
import extramf


def above(mf):
    ''' A.1: above(mf)=0 if x<x_max, 1-mf if x>= x_max; cf below'''
    max_pos = np.argmax(mf)
    new_mf = 1 - mf
    new_mf[:max_pos] = 0
    return new_mf


def any_of(mf):
    '''A.2: any(mf) = 1.. ('any' is a Python built-in)'''
    return np.ones_like(mf)


def below(mf):
    '''A.3: below(mf) = 0 if x>x_max, 1-mf(x) if x<x_max; cf above'''
    max_pos = np.argmax(mf)
    new_mf = 1 - mf
    new_mf[max_pos:] = 0
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


# List of all hedges, maps name to function
_IEEE_HEDGES = {
    'above':        above,
    'any':          any_of,
    'below':        below,
    'extremely':    extremely,
    'intensify':    intensify,
    'more_or_less': more_or_less,
    'norm':         norm,
    'not':          is_not,
    'plus':         plus,
    'seldom':       seldom,
    'slightly':     slightly,
    'somewhat':     somewhat,
    'very':         very,
}


def test_all_hedges(y):
    results = []
    hedgenames = sorted(all_hedges.keys())  # Want them in alphabetical order
    for name in hedgenames:
        func = all_hedges[name]
        results.append(func(y))
    return (hedgenames, results)


if __name__ == '__main__':
    x = np.arange(0, 100)
    y = skmemb.gaussmf(x, 50, 15)
    (titles, data) = test_all_hedges(y)
    extramf.visualise_all(x, [y]+data, ['original (gaussian)']+titles)
