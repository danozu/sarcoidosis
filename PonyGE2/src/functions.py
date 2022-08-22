# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:21:05 2020

@author: allan
"""

import numpy as np

class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args):
        return self.function(*args)


def _add(a, b, x):
    return np.add(a,b)

def _sub(a, b, x):
    return np.subtract(a,b)

def _mul(a, b, x):
    return np.multiply(a,b)

def _protected_division(a, b, x):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(b) > 0.001, np.divide(a, b), 1.)

def _protected_sqrt(a, b, x):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(b))

def _neg(a, b, x):
    return np.negative(b)

def _abs(a, b, x):
    return np.abs(b)

def _WA(a,b,x):
    return x*a+(1-x)*b

def _OWA(a,b,x):
    return x*max(b,a)+(1-x)*min(b,a)    

def _minimo(a, b, x):
    return np.minimum(a,b)

def _maximo(a, b, x):
    return np.maximum(a,b)
    
def _dilator(a, b, x):
    return b**0.5

def _concentrator(a, b, x):
    return b**2





add2 = _Function(function=_add, name='add')
sub2 = _Function(function=_sub, name='sub')
mul2 = _Function(function=_mul, name='mul')
div2 = _Function(function=_protected_division, name='div')
sqrt1 = _Function(function=_protected_sqrt, name='sqrt')
neg1 = _Function(function=_neg, name='neg')
abs1 = _Function(function=_abs, name='abs')
WA3 = _Function(function=_WA, name='WA')
OWA3 = _Function(function=_OWA, name='OWA')
min2 = _Function(function=_minimo, name='min')
max2 = _Function(function=_maximo, name='max')
dilator1 = _Function(function=_dilator, name='dilator')
concentrator1 = _Function(function=_concentrator, name='concentrator')

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'abs': abs1,
                 'neg': neg1,
                 'WA': WA3,
                 'OWA': OWA3,
                 'min': min2,
                 'max': max2,
                 'dilator': dilator1,
                 'concentrator': concentrator1
}


