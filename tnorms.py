# -*- coding: utf-8 -*-
'''
    Norms and co-norms based on the definitions in the IEEE standard.
    Each function here returns a FuzzyAggregationMethods object,
    which is really just a pair of functions: (and_func, or_func).
    @author: james.power@mu.ie Created on Fri Aug 10 12:48:30 2018
'''


import numpy as np
from skfuzzy.control.term import FuzzyAggregationMethods



def min_max():
    '''
        A.14: minimum t-norm, A.21: maximum t-conorm
    '''
    return FuzzyAggregationMethods(np.fmin, np.fmax)

def product_sum():
    '''
        A.15: product t-norm, A.21: Probabilistc sum t-conorm
    '''
    and_func = lambda a, b: a*b
    or_func  = lambda a, b: a+b - (a*b)
    return FuzzyAggregationMethods(and_func, or_func)


def bounded():
    '''
        A.16: bounded difference t-norm, A.22: bounded sum t-conorm
    '''
    and_func = lambda a, b: np.fmax(0, a+b - 1)
    or_func  = lambda a, b: np.fmin(1, a+b)
    return FuzzyAggregationMethods(and_func, or_func)

def lucasiewicz():
    '''This is just a synonym for bounded'''
    return bounded()

def drastic():
    '''
        A.17: drastic product t-norm, A.23: drastic sum t-conorm
    '''
    and_func = lambda a, b: b if a == 1 else a if b == 1 else 0
    or_func  = lambda a, b: b if a == 0 else a if b == 0 else 1
    return FuzzyAggregationMethods(np.vectorize(and_func), np.vectorize(or_func))

def einstein():
    '''
        A.18: Einstein product t-norm, A.24: Einstein sum t-conorm
    '''
    and_func = lambda a, b: (a*b) / (2 - (a+b - a*b))
    or_func  = lambda a, b: (a+b) / (1 + a*b)
    return FuzzyAggregationMethods(and_func, or_func)

def hamacher():
    '''
        A.19: Hamacher product t-norm, A.25: Hamacher sum t-conorm.
        Note 1: Product is wrong in IEEE standard (numerator is not a+b).
        Note 2: Both were also missing a divide-by-zero check.
    '''
#    and_func = lambda a, b: ((a+b) / ((a+b) - a*b))
#    or_func =  lambda a, b: (a+b - 2*a*b) / (1 - a*b)
    and_func = lambda a, b: 0 if a == b == 0 else (a*b) / ((a+b) - a*b)
    or_func =  lambda a, b: 1 if a == b == 1 else (a+b - 2*a*b) / (1 - a*b)
    return FuzzyAggregationMethods(np.vectorize(and_func), np.vectorize(or_func))

def nilpotent():
    '''
        A.20: Nilpotent minum t-norm, A.26: Nilpotent maximum t-conorm
    '''
    and_func = lambda a, b: np.fmin(a, b) if a+b > 1 else 0
    or_func =  lambda a, b: np.fmax(a, b) if a+b < 1 else 1
    return FuzzyAggregationMethods(np.vectorize(and_func), np.vectorize(or_func))



def check_classic(normfun):
    '''
        Test that norm and co-norm work like classic and/or for 0,1 inputs.
    '''
    fam = normfun()
    valrange = (0,1)
    for a in valrange:
        for b in valrange:
            try:
                prod = fam.and_func(a,b)
                dual_sum = 1 - fam.or_func(1-a, 1-b)
                if not prod  == (a and b) == dual_sum:
                    print('check_classic', normfun.__name__, (a,b), prod, dual_sum)
            except ZeroDivisionError as e:
                print('check_classic', normfun.__name__, (a,b), 'Divide by zero')


def check_duality(normfun):
    '''
        Run some tests to make sure that the norm and co-norm are duals.
        That is, they obey de Morgan's law: (a and b) = not((not a) or (not b))
    '''
    fam = normfun()
    valrange = np.arange(0.0, 1.1, 0.1)
    for a in valrange:
        for b in valrange:
            try:
                prod = fam.and_func(a,b)
                # Need to round, since e.g. 1-0.7 is not 0.3 otherwise:
                dual_sum = 1 - fam.or_func(round(1-a, 1), round(1-b, 1))
                if not np.isclose(prod, dual_sum):
                    print('check_duality', normfun.__name__, (a,b), prod, dual_sum)
            except ZeroDivisionError as e:
                print('check_duality', normfun.__name__, (a,b), 'Divide by zero')


all_norms = [
    min_max, product_sum, bounded,
    drastic, einstein, hamacher, nilpotent,
]


import matplotlib.pyplot as plt
import skfuzzy.membership as skmemb

def visualise_all(x, y1, y2, all_norms=all_norms):
    '''Plot the norm and conorm for the given sample inputs'''
    ncols = 3
    fig, axes = plt.subplots(nrows=len(all_norms), ncols=ncols, figsize=(8, 9))
    fig.tight_layout()
    fig.subplots_adjust(bottom=-.25)
    for row, normfun in enumerate(all_norms):
        name = normfun.__name__
        fam = normfun()
        for col in range(ncols): # so all have the same (0,1) y-axis
            axes[row][col].set_ylim([-0.05, 1.05])
        axes[row][0].set_title('Sample inputs')
        axes[row][0].plot(x, y1)
        axes[row][0].plot(x, y2)
        axes[row][1].set_title(name + ' norm')
        axes[row][1].plot(x, fam.and_func(y1,y2))
        axes[row][2].set_title(name + ' co-norm')
        axes[row][2].plot(x, fam.or_func(y1,y2))

if __name__ == '__main__':
    for normfun in all_norms:
        check_classic(normfun)
        check_duality(normfun)
    sample_x = np.arange(0, 100)
    sample_y1 = skmemb.trapmf(sample_x, [15, 30, 55, 75])
    sample_y2 = skmemb.trapmf(sample_x, [25, 45, 70, 85])
    visualise_all(sample_x, sample_y1, sample_y2)
