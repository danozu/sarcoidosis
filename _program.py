# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:24:53 2020

@author: allan
"""

from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement
#from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score

from functions import _function_map, _Function
from utils import check_random_state

import random


class _Program(object):

    """A program-like representation of the evolved program.

    """

    def __init__(self,
                 function_set,
                 const_set,
                 n_classes,
                 Q_QF,
                 Q_QT1,
                 Q_QT2_choice,
                 Q_QT2_feature,
                 Q_QT2_other,
                 Q_QR,
                 metric,
                 parsimony_coefficient=0,
                 feature_names=None,
                 program=None):
        self.function_set = function_set
        self.const_set = const_set
        self.n_classes = n_classes
        self.Q_QF = Q_QF
        self.Q_QT1 = Q_QT1
        self.Q_QT2_choice = Q_QT2_choice
        self.Q_QT2_feature = Q_QT2_feature
        self.Q_QT2_other = Q_QT2_other
        self.Q_QR = Q_QR
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.feature_names = feature_names
        self.program = program
        
        if self.program is None:
            self.program = self.build_program()
        
        self.raw_fitness_ = None
        self.fitness_ = None
        self.valid_ = 0
        self.features_ = None
        
    def raw_fitness_(self):
        return self.raw_fitness_

    def features_(self):
        return self.features_
    
    def observation(self,
                   p):
        """
        Observation of function or terminal token.
        The input is a set of probabilities.
        The output is the observed position.
        For example, if the input has four probabilities, the output can be 0, 1, 2 or 3.
        """
        length = len(p)
        p_ = [0] * length
        p_[0] = p[0]
    
        for i in range(length - 1):  
            p_[i + 1] = p_[i] + p[i + 1] # from 1 to tamanho-1
    
        r = random.random()
    
        if r <= p_[0]:
            Tr = 0
    
        for i in range(length - 1):
            if r > p_[i] and r <= p_[i + 1]:
                Tr = i + 1
        
        return Tr
     
    def build_program(self):
        """
        Function that builds a program (individual), based on the observation of tokens according to the probability 
        distributions of that individual.
        """
        
        L, _, n_positions = np.shape(self.Q_QT1) 
        n_features = len(self.Q_QT2_feature)
        
        program = [[[] for _ in range(4)] for _ in range(L)]
        
        for j in range(L): #preenche cada gene
            tokenF = self.observation(self.Q_QF[j])#, random_state) #observa TF, resultando em 0 (nop), 1 (WA), 2 (OWA), 3 (min), 4 (max), 5 (diluidor) ou 6 (concentrador)
            tokenT1 = self.observation(self.Q_QT1[j,tokenF])#, random_state) #observa TT1
            choiceT2 = self.observation(self.Q_QT2_choice[j,tokenF])#, random_state)
            if choiceT2 == 0:
                feature = self.observation(self.Q_QT2_feature)#, random_state)
                tokenT2 = n_positions + feature #posição do atributo escolhido no vetor
            elif choiceT2 == 1:
                other = self.observation(self.Q_QT2_other[j,tokenF])#, random_state)
                if other < n_positions: #foi escolhida uma das posições iniciais. Por ex, se há cinco posições possíveis, estas são 0, 1, 2, 3 e 4
                    tokenT2 = other
                else: #foi escolhida uma constante, que está nas últimas posições do vetor
                    tokenT2 = n_features + other
                    #por ex, se há 4 posições, 6 argumentos e 3 constantes, o vetor tem posições de 0 a 12
                    #outroEscolhido pode ser de 0 a 7
                    #se for 0, 1, 2 ou 3, é uma posição inicial, e assume esse valor no veotr
                    #se for 4, 5 ou 6 é uma constante
                    #aí soma com 6 para assumir as posições 10, 11 ou 12, que são as últimas do vetor
                
            tokenR = 0
            if tokenF == 1 or tokenF == 2:
                tokenR = self.observation(self.Q_QR[j,tokenF-1])#, random_state)
            program[j][0:4] = tokenF, tokenT1, tokenT2, tokenR
                
        return program
     
    def execute(self, X):
        """
        X is a matrix with n_samples lines and n_features columns.
        The function returns the result of executing the program on X.
        """
        
        L, _, n_positions = np.shape(self.Q_QT1) #L, numFuncoes, numPosicoes
        n_samples, n_features = np.shape(X)
        n_const = len(self.const_set)
        
        R = np.zeros([n_positions+n_features+n_const], dtype=float)
        
        for i in range(n_const): 
            R[n_positions+n_features+i] = self.const_set[i]
        
        data = X.copy()
    
        results = np.empty([n_samples], dtype=float)
        
        for i in range(n_samples):
            #Primeiro, limpa-se possível sujeira nos registradores de posição
            for j in range(n_positions):
                R[j] = 0
            #Os registradores R recebem os atributos nas posições numPosicoes a (numPosicoes+qtdeAtributos)
            R[n_positions:(n_positions+n_features)] = data[i]
            for j in range(L):
                TF = self.program[j][0]
                TT1 = self.program[j][1]
                TT2 = self.program[j][2]
                TR = self.program[j][3] + n_positions + n_features
                
                if TF == 0:
                    pass
                else:
                    function_name = self.function_set[TF-1]
                    R[TT1] = _function_map[function_name](R[TT1],R[TT2],R[TR])
            #O resultado da execução do programa termina nos registradores 0 a (numResultados-1)
            results[i] = R[0]
   
        return results
    
    def raw_fitness(self, results, y, sample_weight=None):
        """
        results : the result using the function execute(X)
        y : labels
        """
        #Verifica se uma das colunas com os resultados é uma constante. Se for, anula o fitness
        const_verify = len(set(results))
        if const_verify == 1:
            if self.metric.greater_is_better:
                return 0
            else:
                return 1
        else:
            raw_fitness = self.metric(y, results, sample_weight)
            
            return raw_fitness
            
    def count_lines(self):
        L, _, _ = np.shape(self.Q_QT1) #L, numFuncoes, numPosicoes
        valid_lines = 0
        for i in range(L):
            if self.program[i][0] != 0: #linhas diferentes de 'NOP()'
                valid_lines += 1
        return valid_lines
    
    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
            
        L, _, _ = np.shape(self.Q_QT1) #L, numFuncoes, numPosicoes
        
        valid_lines = 0
        for i in range(L):
            if self.program[i][0] != 0: #linhas diferentes de 'NOP()'
                valid_lines += 1
        
        penalty = parsimony_coefficient * valid_lines * self.metric.sign
        return self.raw_fitness_ - penalty
    