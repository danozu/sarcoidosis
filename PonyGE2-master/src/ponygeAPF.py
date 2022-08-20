#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

import csv

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from fuzzify import *

#-------------------------
#Criando csv para utilizar
#-------------------------

data = pd.read_csv('C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/sarcoidose/data/sr.csv')

dominio = matrixDomain(data)
datapd = fuzzifyDataFrame(data,3,dominio) #16*3 +2 colunas
l, cfuzzificadas = np.shape(datapd) #cfuzzificadas=50

cols = list(datapd.columns)
cols.pop()

X = datapd.iloc[:,0:(cfuzzificadas-2)].values
y = datapd.iloc[:,(cfuzzificadas-1)].values

#for i in range(l):
#    if y[i] == 0:
#        y[i] = -1 #é da classe 0
#    else:
#        y[i] = 1 #não é da classe 0

#----------------
#Naive classifier
#----------------

#Verificando o equilíbrio entre os conjuntos de treino e validação
D_clf = DummyClassifier(strategy='constant', constant=0)#-1)

p1Tr = 0
p1T = 1

while abs(p1Tr-p1T) > 0.13:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lTreino = len(X_train)
    lTeste = len(X_test)    
    
    D_clf = D_clf.fit(X_train, y_train)
    p1Tr = D_clf.class_prior_[0] #probabilidade da classe 0
    
    D_clf = D_clf.fit(X_test, y_test)
    p1T = D_clf.class_prior_[0] #probabilidade da classe 0

Treino = np.empty([lTreino,cfuzzificadas-1], dtype=float)
Teste = np.empty([lTeste,cfuzzificadas-1], dtype=float)

Treino[:,0:(cfuzzificadas-2)] = X_train
Treino[:,cfuzzificadas-2] = y_train
Teste[:,0:(cfuzzificadas-2)] = X_test
Teste[:,cfuzzificadas-2] = y_test

#c = csv.writer(open("MEUARQUIVO.csv", "wb"))
pd.DataFrame(Treino).to_csv("C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/PonyGE2-master/datasets/SarcoidoseFuzzy/Train1.csv", sep=" ", header=cols, index=None)
pd.DataFrame(Teste).to_csv("C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/PonyGE2-master/datasets/SarcoidoseFuzzy/Test1.csv", sep=" ", header=cols, index=None)


#-------------
#Executando GE
#-------------

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params,load_params#,
import sys

#from pgliq2 import WA, OWA, minimo, maximo, diluidor, concentrador

def mane():
    """ Run program """

    # Run evolution
    individuals = params['SEARCH_LOOP']()
    
    # Print final review
    get_stats(individuals, end=True)


if __name__ == "__main__":
    
    load_params('C:/Users/PICHAU/Dropbox/Mestrado - PEL/Pesquisa/PonyGE2-master/parameters/classification.txt')
    set_params (sys.argv[1:])  # exclude the ponyge.py arg itself
    mane()


