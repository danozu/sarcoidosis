# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:12:55 2020

@author: allan
"""

from utilities.algorithm.general import check_python_version
check_python_version()

import numpy as np
import pandas as pd
import math
from sklearn.utils.validation import check_X_y#, check_array
from sklearn.utils.multiclass import check_classification_targets

from algorithm.parameters import params, set_params,load_params
from utilities.fitness.math_functions import add, mul, sub, pdiv, WA, OWA, minimum, maximum, dilator, concentrator

import sys

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import random
__all__ = ['ponyge']

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 CROSSOVER_PROBABILITY=0.8,
                 MUTATION_PROBABILITY=0.01,
                 GENERATIONS = 10,
                 POPULATION_SIZE=10,
                 MAX_INIT_TREE_DEPTH=10,
                 TOURNAMENT_SIZE=2,
                 MAX_TREE_DEPTH=17,
                 RANDOM_SEED=7,
                 ):
        self.CROSSOVER_PROBABILITY = CROSSOVER_PROBABILITY
        self.MUTATION_PROBABILITY = MUTATION_PROBABILITY
        self.GENERATIONS = GENERATIONS
        self.POPULATION_SIZE = POPULATION_SIZE
        self.MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH
        self.TOURNAMENT_SIZE = TOURNAMENT_SIZE
        self.MAX_TREE_DEPTH = MAX_TREE_DEPTH
        self.RANDOM_SEED = RANDOM_SEED


    def fit(self, X, y):
        """
        """
        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)

            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes = len(self.classes_)

        else:
            X, y = check_X_y(X, y, y_numeric=True)
            
        l, c = np.shape(X)
        self.n_features_ = c

        i = random.randint(0, 1e10)

        data = np.empty([l,c+1], dtype=float)
        data[:,0:c] = X
        data[:,c] = y
        
        head = []
        for j in range(c):
            head.append('x'+str(j))
        head.append('class')

        pd.DataFrame(data).to_csv(r"../datasets/Sarcoidose/Train" + str(i) + ".csv", header=head, sep=" ", index=None)
        pd.DataFrame(data).to_csv(r"../datasets/Sarcoidose/Test" + str(i) + ".csv", header=head, sep=" ", index=None)
        #pd.DataFrame(data).to_csv(r"PonyGE2/datasets/Sarcoidose/Train" + str(i) + ".csv", header=head, sep=" ", index=None)
        #pd.DataFrame(data).to_csv(r"PonyGE2/datasets/Sarcoidose/Test" + str(i) + ".csv", header=head, sep=" ", index=None)

        #load_params(r'PonyGE2/parameters/classification.txt')
        load_params(r'../parameters/classification.txt')

        params['CROSSOVER_PROBABILITY'] = self.CROSSOVER_PROBABILITY
        params['MUTATION_PROBABILITY'] = self.MUTATION_PROBABILITY        
        params['POPULATION_SIZE'] = self.POPULATION_SIZE
        params['GENERATIONS'] = self.GENERATIONS
        params['MAX_INIT_TREE_DEPTH'] = self.MAX_INIT_TREE_DEPTH
        params['TOURNAMENT_SIZE'] = self.TOURNAMENT_SIZE
        params['MAX_TREE_DEPTH'] = self.MAX_TREE_DEPTH
        params['RANDOM_SEED'] = self.RANDOM_SEED
        params['DATASET_TRAIN'] = 'Sarcoidose/Train'+ str(i) + '.csv'
        params['DATASET_TEST'] = 'Sarcoidose/Test'+ str(i) + '.csv'
        params['SAVE_PLOTS'] = False
        params['CACHE'] = False
        params['SILENT'] = True
        params['INITIALISATION'] = 'PI_grow'

#        params['GRAMMAR_FILE'] = 'supervised_learning/Sarcoidose'+ str(i) + '.bnf'

        set_params (sys.argv[1:])  # exclude the ponyge.py arg itself
        
        self.individuals = params['SEARCH_LOOP']()
        
        self.best_individual = max(self.individuals)
        
        self.phenotype = self.best_individual.phenotype
        
        return self
 
class ponyge(BaseSymbolic, ClassifierMixin):
    """
    """
    def __init__(self,
                 CROSSOVER_PROBABILITY=0.8,
                 MUTATION_PROBABILITY=0.01,
                 GENERATIONS = 10,
                 POPULATION_SIZE=10,
                 MAX_INIT_TREE_DEPTH=10,
                 TOURNAMENT_SIZE=2,
                 MAX_TREE_DEPTH=17,
                 RANDOM_SEED=7,
                 ):
         super(ponyge, self).__init__(
             CROSSOVER_PROBABILITY = CROSSOVER_PROBABILITY,
             MUTATION_PROBABILITY = MUTATION_PROBABILITY,
             GENERATIONS = GENERATIONS,
             POPULATION_SIZE = POPULATION_SIZE,
             MAX_INIT_TREE_DEPTH = MAX_INIT_TREE_DEPTH,
             TOURNAMENT_SIZE = 2,
             MAX_TREE_DEPTH = MAX_TREE_DEPTH,
             RANDOM_SEED = RANDOM_SEED)

    def predict_proba(self, X):
        """Predict probabilities on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        proba : array, shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.

        """

        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))
            
        l, c = np.shape(X)
        
        yhat = np.zeros([l], dtype=float)

        for i in range(l):
            converter = {
                'add': add,
                'mul': mul,
                'sub': sub,
                'pdiv': pdiv,
                'WA': WA,
                'OWA': OWA,
                'minimum': minimum,
                'maximum': maximum,
                'dilator': dilator,
                'concentrator': concentrator,
                'x': X[i]
            }
            yhat[i] = eval(self.best_individual.phenotype, {}, converter)

        fuzzy = 1
        #if max(yhat) <= 1 and min(yhat) >= 0:
        if fuzzy == 1:
            proba = np.vstack([1 - yhat, yhat]).T
        else:
            sigmoid = np.zeros([len(yhat)], dtype=float)
            for i in range(len(yhat)):
                try:
                    sigmoid[i] = 1 / (1 + math.exp(-yhat[i]))
                except OverflowError:
                    s = np.sign(yhat[i])*float("inf")
                    sigmoid[i] = 1 / (1 + math.exp(-s))
            proba = np.vstack([1 - sigmoid, sigmoid]).T

        return proba

    def predict(self, X):
        """Predict classes on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples,]
            The predicted classes of the input samples.

        """
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)      