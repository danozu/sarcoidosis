# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 18:20:48 2020

@author: allan
"""

import itertools
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import compute_sample_weight
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import check_classification_targets

from _program import _Program
from fitness import _fitness_map, _Fitness
from functions import _function_map, _Function
from utils import check_random_state

from sklearn.model_selection import KFold

import random

__all__ = ['FPTClassifier']

def operadorP(q, i, s):
    """Function for updating the probabilities.
    - q: probabilities set
    - i: position of q to increment
    - s: increment step
    At the end, a normalization is performed.
    """
    p = q
    l = len(q)
    p[i] = p[i] + s*(1-p[i])
    total = 0
    for j in range(l):
        total = total + p[j]
    #normalization
    for j in range(l):
        p[j] = p[j]/total
    return p

class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 population_size=6,
                 individual_length=32,
                 step = 0.004,
                 generations=50000,
                 stopping_criteria=0.0,
                 gwi_max=10000, #maximum number of generations without improvement
                 const_set=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                 function_set=('WA', 'OWA', 'min', 'max', 'dilator', 'concentrator'),
                 metric='roc_auc',
                 parsimony_coefficient=0.001,
                 n_positions=6,
                 feature_names=None,
                 random_state=None
                 #n_jobs=1,
                 #verbose=0,
                 ):

        self.population_size = population_size
        self.individual_length = individual_length
        self.step = step
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.gwi_max = gwi_max
        self.const_set = const_set
        self.function_set = function_set
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.n_positions = n_positions
        self.feature_names = feature_names
       # self.n_jobs = n_jobs
       # self.verbose = verbose
        self.random_state = random_state
        


    def fit(self, X, y, n_folds=5):
        """
        """
        random.seed(self.random_state)
        ob = 0
        #It is necessary to shuffle, because in the order it will be possible to have sets with only one class
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        if isinstance(self, ClassifierMixin):
            X, y = check_X_y(X, y, y_numeric=False)
            check_classification_targets(y)

            self.classes_, y = np.unique(y, return_inverse=True)
            self.n_classes = len(self.classes_)

        else:
            X, y = check_X_y(X, y, y_numeric=True)

        _, self.n_features_ = X.shape

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'f1' and self.metric != 'roc_auc':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = _fitness_map[self.metric]

        if self.feature_names is not None:
            if self.n_features_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_, len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, str):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))
        
        step_2 = self.step/2
        step_3 = self.step/50
#        gBest = 0 #generation of the best individual
        gwi = 0 #initial number of generations without improvement
        
        n_functions = len(self.function_set) + 1 #mais o NOP
        n_constants = len(self.const_set) #0.1 to 0.9
        n_arguments = self.n_features_ #number of possible features to be chosen
#        numPosicoes = 3*self.n_classes #número das primeiras posições que receberão resultados de operações
#        self.n_positions
               
        M = self.population_size
        L = self.individual_length
        
        #Matrizes com as probabilidades de cada função, argumento ou valor de x ocorrer em cada linha de cada indivíduo
        Q_QF = np.empty([M,L,n_functions], dtype=float)
        Q_QT1 = np.empty([M,L,n_functions,self.n_positions], dtype=float) #TT1
        Q_QT2_choice = np.empty([M,L,n_functions,2], dtype=float) #matriz com a probabilidade do T2 ser um atributo (posição 0) ou outro (posição 1)
        Q_QT2_feature = np.empty([M,n_arguments], dtype=float)
        Q_QT2_other = np.empty([M,L,n_functions,self.n_positions+n_constants], dtype=float) #TT2
        Q_QR = np.empty([M,L,2,9], dtype=float) #probabilidades das constantes usadas por WA e OWA (0.1 a 0.9)
        
        #Matrizes com o melhor indivíduo (validado)
        Cm = np.empty([L,4], dtype=int) 
        
        if self._metric.greater_is_better:
            fitness_Cm = 0 #fitness of the best individual at the moment (train)
            fitness_val_Cm = 0 #validation
        else:
            fitness_Cm = 1
            fitness_val_Cm = 1
        
        for gen in range(self.generations):
            #Inicializa as probabilidades na primeira geração e quando gwi atingir o valor máximo (neste caso somente se o número de 
            #gerações estiver abaixo de 90% do total, para dar tempo das probabilidades convergirem, já que estou tomando alguma 
            #interpretabilidade a partir do resultado do seletor de atributos)
            if gen == 0 or (gwi == self.gwi_max and gen < 0.9*self.generations):
                #probabilidades iniciais das funções
                for i in range(M):
                    for j in range(L):
                        for k in range(n_functions):
                            if k == 0:
                                Q_QF[i,j,k] = 0.9 #função NOP
                            else:
                                Q_QF[i,j,k] = 0.1/(n_functions-1) #outras funções
                     
                #probabilidades iniciais dos terminais
                for i in range(M):
                    for j in range(L):
                        for k in range(n_functions):
                            for l in range(self.n_positions): #tokens para o argumento i
                                Q_QT1[i,j,k,l] = 1/(self.n_positions)
                            for l in range(2):
                                Q_QT2_choice[i,j,k,l] = 0.5 #probabilidade de escolha de atributo (posição 0) ou outro (posição 1)
                            for l in range(self.n_positions+n_constants): #tokens para o argumento j
                                Q_QT2_other[i,j,k,l] = 1/(self.n_positions+n_constants)
                
                #seletor de atributos iniciado com todas as probabilidades iguais
                for i in range(M):
                    for j in range(n_arguments):
                        Q_QT2_feature[i,j] = 1/(n_arguments)
                
                for i in range(M):
                    for j in range(L):
                        for k in range(2):
                            for l in range(9):
                                Q_QR[i,j,k,l] = 1/9
                if gwi == self.gwi_max:
                   #TODO
                    #O indivíduo C 1 da população clássica recebe uma cópia de C m como semente.
                    gwi = 0
                    #Elimina a população atual, deixando somente uma cópia do melhor indivíduo até o momento
                    for i in range(len(population)):
                        population.pop()
                    population.append(self._program)

            programs = []
            for i in range(M): #preenche cada indivíduo
                program = _Program(function_set=self.function_set,
                                   const_set=self.const_set,
                                   n_classes=self.n_classes,
                                   Q_QF=Q_QF[i],
                                   Q_QT1=Q_QT1[i],
                                   Q_QT2_choice=Q_QT2_choice[i],
                                   Q_QT2_feature=Q_QT2_feature[i],
                                   Q_QT2_other=Q_QT2_other[i],
                                   Q_QR=Q_QR[i],
                                   metric=self._metric,
                                   program=None)
                results = program.execute(X)
                #Toma a média dos resultados nas pastas de treino
                sum_fitness = 0
                for train_index, valid_index in kf.split(X): 
                    sum_fitness += program.raw_fitness(results[train_index], y[train_index])
                program.raw_fitness_ = sum_fitness/n_folds
                programs.append(program)
            
            if gen==0:
                population = programs.copy() #M indivíduos
            else:
                for p in programs:
                    population.append(p) #2M indivíduos
            
            #Checa programas com o mesmo fitness e anula o maior
            if len(population) == 2*M:
                for i in range(M):
                    for j in range(M):
                        if population[i].raw_fitness_ == population[j+M].raw_fitness_:
                            n_lines_i = population[i].count_lines()
                            n_lines_j = population[j+M].count_lines()
                            if n_lines_i > n_lines_j:
                                population[i].raw_fitness_ = 0 if self._metric.greater_is_better else 1
                            elif n_lines_j > n_lines_i:
                                population[j+M].raw_fitness_ = 0 if self._metric.greater_is_better else 1
                            #Se forem do mesmo tamanho, não anula nenhum
                                
            #a partir da segunda geração (após reset), ordena os 2M indivíduos pelo fitness e elimina os M piores
            if self._metric.greater_is_better:
                #ordenando do maior para o menor
                population = sorted(population, key = _Program.raw_fitness_, reverse=True) 
            else:
                #ordenando do menor para o maior
                population = sorted(population, key = _Program.raw_fitness_, reverse=False) 
            
            for i in range(len(population)-M):
                #Keep only the top M
                population.pop()
            
            #Updating probability distributions
            #Individual = population[i].program, so i is from 0 to M-1, because there are M individuals
            #Each individual is a list Lx4
            for i in range(M):  # Updating probabilities
                if population[i].raw_fitness_ == 0:
                    pass
                else:
                    for j in range(L):
                        # Updating probabilities of functions
                        Q_QF[i,j] = operadorP(Q_QF[i,j], population[i].program[j][0], self.step)
            for i in range(M): # Updating probabilities of terminals
                if population[i].raw_fitness_ == 0:
                    pass
                else:
                    for j in range(L):
                        for k in range(n_functions):
                            if k == population[i].program[j][0]: #if the checked function was observed
                                if k != 0: #if the observed function is not NOP
                                    Q_QT1[i,j,k] = operadorP(Q_QT1[i,j,k],population[i].program[j][1],self.step)
                                    if population[i].program[j][2] < self.n_positions or population[i].program[j][2] >= (self.n_positions+n_arguments):
                                        #por ex, se há 4 posições, 6 argumentos e 3 constantes, TT2 tem valores entre 0 e 12
                                        #se for 0, 1, 2, 3 ou 10, 11, 12 é um outro
                                        Q_QT2_choice[i,j,k] = operadorP(Q_QT2_choice[i,j,k],1,self.step) #posição 1 é dos outros
                                        Q_QT2_other[i,j,k] = operadorP(Q_QT2_other[i,j,k],population[i].program[j][2],step_2) if population[i].program[j][2] < self.n_positions else operadorP(Q_QT2_other[i,j,k],population[i].program[j][2]-n_arguments,step_2)
                                    else:                        
                                        #se for 4, 5, 6, 7, 8, 9 é um atributo
                                        Q_QT2_choice[i,j,k] = operadorP(Q_QT2_choice[i,j,k],0,self.step) #posição 0 é dos atributos
                                        Q_QT2_feature[i] = operadorP(Q_QT2_feature[i],population[i].program[j][2]-self.n_positions,step_3)
                                #TODO corrigir essa parte para localizar antes a localização de WA ou OWA        
                                if k == 1 or k == 2: #só atualiza as probabilidades de R se observar WA ou OWA
                                    Q_QR[i,j,k-1] = operadorP(Q_QR[i,j,k-1],population[i].program[j][3],self.step) 
                            
            #Validating the best individual
            if self._metric.greater_is_better:
                if population[0].raw_fitness_ > fitness_Cm and population[0].valid_ == 0:
                    population[0].valid_ = 1
                    results = population[0].execute(X)
                    sum_fitness = 0
                    for train_index, valid_index in kf.split(X): 
                        sum_fitness += population[0].raw_fitness(results[valid_index], y[valid_index])
                    fitness_valid = sum_fitness/n_folds
                    if fitness_valid > fitness_val_Cm:
                        fitness_Cm = population[0].raw_fitness_
                        fitness_val_Cm = fitness_valid
                        Cm = population[0].program
                        gwi = 0
#                        gBest = gen
                        # Find the best individual in the final generation
                        self._program = population[0]
                    else:
                        gwi += 1
                else:
                    gwi += 1
            else: #TODO
                pass                
        self.features_ = Q_QT2_feature[0]
                
        #Final program without lines "NOP()"
        numNOP = 0
        for i in range(L):
            if self._program.program[i][0] == 0:  # function "NOP()"
                numNOP += 1

        matrix_program = np.empty([L - numNOP, 4], dtype=int)

        j = 0
        for i in range(L):
            if self._program.program[i][0] != 0:
                matrix_program[j, :] = self._program.program[i][:]
                j += 1

        FinalIndividual = []
        
        #Keep the program without introns in FinalIndividual and after in self._final_genotype 
        separaAtributos(FinalIndividual, matrix_program, 0)
        self._final_genotype = FinalIndividual
        
        #Keep the program written in LISP and his length
        self._feature_Lisp = []
        self._feature_length = []
        self._feature_length.append(len(FinalIndividual))
        self._feature_Lisp.append(imprimeLisp(FinalIndividual, self.n_positions, self.feature_names, self.const_set))

        return self
 
def separaAtributos(atributoAconstruir,programa,indice):
    """Função que recebe um programa e um índice e retorna um programa de tamanho menor ou igual
    somente com as linhas que influenciam o registrador final com aquele índice."""

    parcial = [] #lista que receberá as posições que influenciam o resultado
    i = len(programa) - 1 #índice da última linha do programa
    while i >= 0: #loop executado da última até a primeira linha do programa
        if programa[i][1] == indice or programa[i][1] in parcial:
            atributoAconstruir.insert(0, programa[i])
            if programa[i][2] != indice and programa[i][2] not in parcial:
                parcial.append(programa[i][2])
        i -= 1
    return


def imprimeLisp(programaAtributo, numPosicoes, nomeAtributos, constantes, fuzzy=1):
    """Função que retorna o programa em linguagem LISP."""
    numAtributos = len(nomeAtributos)
    numConstantes = len(constantes)
    nomePosicoes = []
    for i in range(numPosicoes):
        nomePosicoes.append(str(i))
    for i in range(numAtributos):
        nomePosicoes.append(nomeAtributos[i])
    for i in range(numConstantes):
        nomePosicoes.append(str(constantes[i]))

    programa = ''  # programa final
    listaProg = []  # lista que receberá os elementos que formarão o programa
    i = len(programaAtributo) - 1
    if fuzzy == 0:
        if i >= 0:
            # Primeiras inclusões na lista
            listaProg.insert(0, ')')
            listaProg.insert(0, str(programaAtributo[i][2]))
            listaProg.insert(0, ',')
            listaProg.insert(0, str(programaAtributo[i][1]))
            if programaAtributo[i][0] == 1:
                listaProg.insert(0, 'add(')
            if programaAtributo[i][0] == 2:
                listaProg.insert(0, 'sub(')
            if programaAtributo[i][0] == 3:
                listaProg.insert(0, 'mul(')
            if programaAtributo[i][0] == 4:
                listaProg.insert(0, 'div(')
            i -= 1
            while i >= 0:  # loop executado da última até a primeira linha do programa
                # Verificando em que posição (ou posições) o índice do atual elemento já apareceu
                posicoes = []
                for j in range(len(listaProg)):
                    if str(programaAtributo[i][1]) == listaProg[j]:
                        posicoes.append(j)
                l = len(posicoes)  # será igual a 1 ou mais, dependendo de quantas vezes o índice apareceu
                del (listaProg[posicoes[0]])
                listaProg.insert(posicoes[0], ')')
                listaProg.insert(posicoes[0], str(programaAtributo[i][2]))
                listaProg.insert(posicoes[0], ',')
                listaProg.insert(posicoes[0], str(programaAtributo[i][1]))
                if programaAtributo[i][0] == 1:
                    listaProg.insert(posicoes[0], 'add(')
                elif programaAtributo[i][0] == 2:
                    listaProg.insert(posicoes[0], 'sub(')
                elif programaAtributo[i][0] == 3:
                    listaProg.insert(posicoes[0], 'mul(')
                elif programaAtributo[i][0] == 4:
                    listaProg.insert(posicoes[0], 'div(')
                if l > 1:
                    j = 1
                    while j < l:
                        posicoes[
                            j] += 4 * j  # por conta da mudança de posiçao ocorrida nas operações com a posição anterior e é multiplicado por j, pois a cada iteração, todas as posições seguintes devem ser incrementadas
                        del (listaProg[posicoes[j]])
                        listaProg.insert(posicoes[j], ')')
                        listaProg.insert(posicoes[j], str(programaAtributo[i][2]))
                        listaProg.insert(posicoes[j], ',')
                        listaProg.insert(posicoes[j], str(programaAtributo[i][1]))
                        if programaAtributo[i][0] == 1:
                            listaProg.insert(posicoes[j], 'add(')
                        elif programaAtributo[i][0] == 2:
                            listaProg.insert(posicoes[j], 'sub(')
                        elif programaAtributo[i][0] == 3:
                            listaProg.insert(posicoes[j], 'mul(')
                        elif programaAtributo[i][0] == 4:
                            listaProg.insert(posicoes[j], 'div(')
                        j += 1

                i -= 1

            for i in range(len(listaProg)):
                for j in range(len(nomePosicoes)):
                    if listaProg[i] == str(j):
                        listaProg[i] = nomePosicoes[j]

            for i in range(len(listaProg)):
                # substitui as posições iniciais pelo valor com que elas são inicializadas (zero)
                for j in range(numPosicoes):
                    if listaProg[i] == str(j):
                        listaProg[i] = str(0)
                programa += listaProg[i]

    if fuzzy == 1:
        if i >= 0:
            # Primeiras inclusões na lista
            listaProg.insert(0, ')')
            if programaAtributo[i][0] == 1 or programaAtributo[i][0] == 2:  # funções WA e OWA têm três argumentos
                listaProg.insert(0, str(constantes[programaAtributo[i][3]]))
                listaProg.insert(0, ',')
            listaProg.insert(0, str(programaAtributo[i][2]))
            if programaAtributo[i][0] != 5 and programaAtributo[i][
                0] != 6:  # funções concentrador e diluidor têm um argumento
                listaProg.insert(0, ',')
                listaProg.insert(0, str(programaAtributo[i][1]))
            if programaAtributo[i][0] == 1:
                listaProg.insert(0, 'WA(')
            elif programaAtributo[i][0] == 2:
                listaProg.insert(0, 'OWA(')
            elif programaAtributo[i][0] == 3:
                listaProg.insert(0, 'minimo(')
            elif programaAtributo[i][0] == 4:
                listaProg.insert(0, 'maximo(')
            elif programaAtributo[i][0] == 5:
                listaProg.insert(0, 'diluidor(')
            elif programaAtributo[i][0] == 6:
                listaProg.insert(0, 'concentrador(')
            i -= 1
            while i >= 0:  # loop executado da última até a primeira linha do programa
                # Verificando em que posição (ou posições) o índice do atual elemento já apareceu
                posicoes = []
                for j in range(len(listaProg)):
                    if str(programaAtributo[i][1]) == listaProg[j]:
                        posicoes.append(j)
                l = len(posicoes)  # será igual a 1 ou mais, dependendo de quantas vezes o índice apareceu
                if l > 0:
                    del (listaProg[posicoes[0]])
                    desloca = -1
                    listaProg.insert(posicoes[0], ')')
                    desloca += 1
                    if programaAtributo[i][0] == 1 or programaAtributo[i][
                        0] == 2:  # funções WA e OWA têm três argumentos
                        listaProg.insert(posicoes[0], str(constantes[programaAtributo[i][3]]))
                        desloca += 1
                        listaProg.insert(posicoes[0], ',')
                        desloca += 1
                    listaProg.insert(posicoes[0], str(programaAtributo[i][2]))
                    desloca += 1
                    if programaAtributo[i][0] != 5 and programaAtributo[i][
                        0] != 6:  # funções concentrador e diluidor têm um argumento
                        listaProg.insert(posicoes[0], ',')
                        desloca += 1
                        listaProg.insert(posicoes[0], str(programaAtributo[i][1]))
                        desloca += 1
                    if programaAtributo[i][0] == 1:
                        listaProg.insert(posicoes[0], 'WA(')
                    elif programaAtributo[i][0] == 2:
                        listaProg.insert(posicoes[0], 'OWA(')
                    elif programaAtributo[i][0] == 3:
                        listaProg.insert(posicoes[0], 'minimo(')
                    elif programaAtributo[i][0] == 4:
                        listaProg.insert(posicoes[0], 'maximo(')
                    elif programaAtributo[i][0] == 5:
                        listaProg.insert(posicoes[0], 'diluidor(')
                    elif programaAtributo[i][0] == 6:
                        listaProg.insert(posicoes[0], 'concentrador(')
                    desloca += 1
                if l > 1:
                    j = 1
                    while j < l:
                        posicoes[
                            j] += desloca  # 4*j por conta da mudança de posiçao ocorrida nas operações com a posição anterior e é multiplicado por j, pois a cada iteração, todas as posições seguintes devem ser incrementadas
                        del (listaProg[posicoes[j]])
                        desloca -= 1
                        listaProg.insert(posicoes[j], ')')
                        desloca += 1
                        if programaAtributo[i][0] == 1 or programaAtributo[i][
                            0] == 2:  # funções WA e OWA têm três argumentos
                            listaProg.insert(posicoes[j], str(constantes[programaAtributo[i][3]]))
                            desloca += 1
                            listaProg.insert(posicoes[j], ',')
                            desloca += 1
                        listaProg.insert(posicoes[j], str(programaAtributo[i][2]))
                        desloca += 1
                        if programaAtributo[i][0] != 5 and programaAtributo[i][
                            0] != 6:  # funções concentrador e diluidor têm um argumento
                            listaProg.insert(posicoes[j], ',')
                            desloca += 1
                            listaProg.insert(posicoes[j], str(programaAtributo[i][1]))
                            desloca += 1
                        if programaAtributo[i][0] == 1:
                            listaProg.insert(posicoes[j], 'WA(')
                        elif programaAtributo[i][0] == 2:
                            listaProg.insert(posicoes[j], 'OWA(')
                        elif programaAtributo[i][0] == 3:
                            listaProg.insert(posicoes[j], 'minimo(')
                        elif programaAtributo[i][0] == 4:
                            listaProg.insert(posicoes[j], 'maximo(')
                        elif programaAtributo[i][0] == 5:
                            listaProg.insert(posicoes[j], 'diluidor(')
                        elif programaAtributo[i][0] == 6:
                            listaProg.insert(posicoes[j], 'concentrador(')
                        desloca += 1
                        j += 1

                i -= 1

            for i in range(len(listaProg)):
                for j in range(len(nomePosicoes)):
                    if listaProg[i] == str(j):
                        listaProg[i] = nomePosicoes[j]

            for i in range(len(listaProg)):
                # substitui as posições iniciais pelo valor com que elas são inicializadas (zero)
                for j in range(numPosicoes):
                    if listaProg[i] == str(j):
                        listaProg[i] = str(0)
                programa += listaProg[i]

    return programa
   
class FPTClassifier(BaseSymbolic, ClassifierMixin):

    """A Genetic Programming symbolic classifier.


    """
    def __init__(self,
              population_size=6,
              individual_length=32,
              step = 0.004,
              generations=50000,
              stopping_criteria=0.0,
              gwi_max=10000, #maximum number of generations without improvement
              const_set=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
              function_set=('WA', 'OWA', 'min', 'max', 'dilator', 'concentrator'),
              metric='roc_auc',
              n_positions=6,
              feature_names=None,
              #n_jobs=1,
              #verbose=0,
              random_state=None):
         super(FPTClassifier, self).__init__(
             population_size = population_size,
             individual_length = individual_length,
             step = step,
             generations = generations,
             stopping_criteria = stopping_criteria,
             gwi_max = gwi_max,
             const_set = const_set,
             function_set = function_set,
             metric = metric,
             n_positions = n_positions,
             feature_names = feature_names,
             #n_jobs = n_jobs,
             #verbose = verbose,
             random_state = random_state)

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
        
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicClassifier not fitted.')
            
        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))

        results = self._program.execute(X)
        
        proba = np.vstack([1 - results, results]).T
        
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
    
    def score(self, y, y_pred, metric):
        return _fitness_map[metric](y, y_pred)
        
        
        
