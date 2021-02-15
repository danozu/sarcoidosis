# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:21:17 2020

@author: allan
"""

import numbers

import numpy as np
#from joblib import wrap_non_picklable_objects
#from scipy.stats import rankdata

from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score



class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)

def _f1(y, results, w=None):
    """Calculate the f1-score."""
    y_pred = np.zeros([len(y)], dtype=float)
    for i in range(len(y)):
        if results[i] <= 0.5: #TODO
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    
    return f1_score(y, y_pred)

def _roc_auc(y, results, w=None):
    
    proba = np.vstack([1 - results, results]).T
    
    #predict_proba
    #y_scores = np.zeros([len(y), len(set(y))], dtype=float) #n_samples=len(X); n_classes=len(set(y))
    #Estava dando erros em conjuntos pequenos que só tinham etiquetas de uma classe
#    y_scores = np.zeros([len(y), 2], dtype=float) #n_samples=len(X); n_classes=len(set(y))
            
#    for i in range(len(y)):
        #Vamos normalizar de forma que a soma das pertinências seja igual a 1
        #Exceto quando todas as pertinências forem zero. 
        #Neste caso, as probabilidades serão todas iguais a zero também.
#        somaPertinencias = sum(results[i,:])
#        if somaPertinencias == 0:
#            y_scores[i] = 0 #todas as colunas da linha i receberão esse valor
#        else:
#            y_scores[i] =  results[i,:] / somaPertinencias
        
    fpr, tpr, threshold = roc_curve(y, proba[:, 1])
    return auc(fpr, tpr)
        
f1 = _Fitness(function=_f1, greater_is_better=True)
roc_auc = _Fitness(function=_roc_auc, greater_is_better=True)

_fitness_map = {'f1': f1,
                'roc_auc': roc_auc
                }
