#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:37:00 2019

@author: allan
"""

import numpy as np
import pandas as pd
import re
import skfuzzy as fuzz

def matrixDomain(DataFrame, train_index):
    
    df = DataFrame.copy() #necessário para que a função não altere o dataframe original
    
    #remove linhas com valores nulos para evitar erros
    df.dropna(inplace = True)  
    
    #Dimensões do dataframe (não necessitamos do número de linhas)
    _, numColumns = np.shape(df) #o número de colunas equivale ao número de variáveis do dataframe
    
    #Cria matriz vazia com duas colunas, cujo número de linhas é igual ao número de variáveis do dataframe
    matrixDomain = np.empty([numColumns,2], dtype=float)
    
    #Seleciona dados reais
    dataReal = df.select_dtypes(include=['float64'])
    _, numColumnsReal = np.shape(dataReal)
    
    #Encontrei erro em diferentes versões, pois em uma aceitam "int", e em outra "int64"
    #Para confirmar os tipos de dados das colunas, fazer df.dtypes
    
    #Seleciona dados categóricos
    dataCategorical = df.select_dtypes(include=['object','int64'])
    _, numColumnsCategorical = np.shape(dataCategorical)
    
    #Vetores com os limites mínimo e máximo dos dados reais
    minimum = np.zeros([numColumnsReal], dtype=float)
    maximum = np.zeros([numColumnsReal], dtype=float)
    array = dataReal.values
    X_train = array[train_index == 1.0]
    for i in range(numColumnsReal):
        minimum[i] = min(X_train[:,i])
        maximum[i] = max(X_train[:,i])
    
    #minimum, maximum = dataReal.min(), dataReal.max()
    
    #Verificação de segurança
    if (numColumnsReal + numColumnsCategorical) != numColumns:
        print("Attention! Presence of discreet non-categorical columns.")
        print("The domain matrix will not be filled.")
        print()
        return
    
    #nomes das variáveis
    nameCol = df.columns #todas as variáveis
    nameColReal = dataReal.columns #variáveis reais
    nameColCategorical = dataCategorical.columns #variáveis categóricas
    
    j, k = 0, 0 #índices das colunas real e categórica, respectivamente
    for i in range(numColumns):
        if j < numColumnsReal: #se ainda há colunas reais a verificar
            if nameCol[i] == nameColReal[j]:
                #preenche a linha i com o mínimo e o máximo da variável real j
                matrixDomain[i][:] = minimum[j], maximum[j]
                j += 1
        if k < numColumnsCategorical: #se ainda há colunas categóricas a verificar
            if nameCol[i] == nameColCategorical[k]:
                matrixDomain[i] = len(dataCategorical[nameColCategorical[k]].unique())
                k += 1 
    
    return matrixDomain

def fuzzifyDataFrame(DataFrame, nSets, matrixDomain):
    
    df = DataFrame.copy() #necessário para que a função não altere o dataframe original
    
    #dimensões do dataframe
    numRows, numColumns = np.shape(df)
    
    #verifica se todas as variáveis estão referenciadas na matriz de domínio
    if np.shape(matrixDomain) == ():
        numVariables = 0
    else:    
        numVariables, _ = np.shape(matrixDomain)
    if numVariables != numColumns:
        print("Domain matrix does not represent all the variables")
        return
    
    #todos os índices do dataframe
    totalIndexes = list(df.index)
    
    #remove valores nulos para evitar erros
    df.dropna(inplace = True) 
    
    numRowsAposRemocao,_ = np.shape(df)
    numLinhasEliminadas = numRows - numRowsAposRemocao
    if numLinhasEliminadas != 0:
        print("Warning: %i lines were deleted because they didn't contain all the attributes" %numLinhasEliminadas)
    
    #índices dos dados que não foram removidos
    validIndexes = list(df.index)
    
    #dimensões do dataframe
    validNumberRows, _ = np.shape(df) #o número de colunas não interessa, pois não mudou
    
    #após este loop, totalIndexes conterá somente os índices que foram removidos
    for i in range(validNumberRows):
        totalIndexes.remove(validIndexes[i])

    #copia o dataframe para outros dois dataframes
    dataReal = df.copy()
    dataCategorical = df.copy()
    
    nonReal = [] #no dataReal serão eliminadas as colunas categóricas
    nonCategorical = [] #no dataCategorical serão eliminadas as colunas não-categóricas
    
    for i in range(numVariables):
        if matrixDomain[i,0] == matrixDomain[i,1]: #dados categóricos
            nonReal.append(i)
        else: #dados não categóricos
            nonCategorical.append(i)
    
    dataReal.drop(dataReal.columns[nonReal], axis=1, inplace=True)
    dataCategorical.drop(dataCategorical.columns[nonCategorical], axis=1, inplace=True)
    
    #dimensões da parte não categórica do dataframe
    #dataReal = df.select_dtypes(include=['float64'])
    _, numColumnsReal = np.shape(dataReal) #o número de linhas não interessa, pois é o mesmo em validNumberRows
        
    #dimensões da parte categórica do dataframe
    #dataCategorical = df.select_dtypes(include=['object'])
    _, numColumnsCategorical = np.shape(dataCategorical)
        
    #nomes das variáveis
    nameCol = df.columns #todos os nomes
    nameColReal = dataReal.columns #nomes das variáveis reais
    nameColCategorical = dataCategorical.columns #nomes das variáveis categóricas
    
    arraySets = np.empty(numColumns, dtype=int)
    #arraySets armazena o número de conjuntos em que cada variável se dividirá
    #Caso a entrada nSets seja um valor inteiro, quer dizer que todas as variáveis não categóricas
    #se  dividirão no mesmo número de conjuntos
    #Caso seja um array, cada posição faz referência a uma coluna com dados no dataframe
    #A posição referente a uma coluna categórica deve conter exatamente o mesmo número de 
    #categorias em que os dados estão divididos
    j, k = 0, 0 #índices dos nomes de colunas com dados reais e categóricos, respectivamente
    if type(nSets) == int:
        if nSets < 2:
            print("Number of sets must be greater than or equal to 2")
            return
        else:
            for i in range(numColumns):
                if numColumnsReal > j: #se ainda há colunas reais a verificar
                    if nameCol[i] == nameColReal[j]:
                        arraySets[i] = nSets
                        j += 1
                if numColumnsCategorical > k: #idem para colunas categóricas
                    if nameCol[i] == nameColCategorical[k]:
                        #se o número de categorias indicado na matriz é realmente o número de categorias em que os dados se dividem
                        if matrixDomain[i,0] == len(dataCategorical[nameColCategorical[k]].unique()):
                            arraySets[i] = matrixDomain[i,0]
                            k += 1
                        else:
                            print('{0}{1}{2}'.format("Number of categories of the variable ",nameCol[i]," is different from that indicated"))
                            return
    else: #se o valor passado foi um vetor
        nSetsSize = len(nSets)
        if numVariables != nSetsSize:
            print("Size of the array nSets must be equal to the number of variables.")
            return
        for i in range(numColumns):
            if matrixDomain[i,0] == matrixDomain[i,1]: #indicação de dados categóricos
                if nSets[i] != matrixDomain[i,0]:
                    print('{0}{1}{2}'.format("Number of categories of the variable ",nameCol[i]," is different from that indicated"))
                    return
            if nSets[i] < 2:
                print("Number of sets must be greater than or equal to 2")
                return
            arraySets[i] = nSets[i]
    
    #é necessário passar os dados do dataframe para uma matriz, pois o dataframe pode ter
    #tido linhas deletadas (por falta de dados), o que dificulta o seu endereçamento
    #Nesta matriz, as posições do dataframe com dados deletados ficarão zeradas
    matrixDataReal = np.zeros([numRows,numColumnsReal], dtype=float)
    
    sumSets = int(sum(arraySets)) #total de conjuntos a dividir as variáveis
    pertinenceMatrix = np.zeros([numRows,sumSets], dtype=float) #matriz de pertinência
    pertinenceMatrixDF = {} #dataframe final
    
    i = 0 #não uso o index para referenciar, porque pode ter linhas deletadas
    for index, row in dataReal.iterrows(): 
        for j in range(numColumnsReal):
            matrixDataReal[validIndexes[i]][j] = row[nameColReal[j]]
        i += 1
    
    #hora de preencher a matriz de pertinência
    actualColumn = 0 #coluna que será preenchida
    actualIndexSets = 0
    for i in range(numColumns):
        if matrixDomain[i,0] == matrixDomain[i,1]: #dados categóricos
            arrayCategories = np.empty(arraySets[i], dtype=str) #arraySets[i] é o número de categorias
            arrayCategories = df[nameCol[i]].unique() #nome de cada categoria da coluna i
            #Se as categorias são números, especialmente 0 e 1, e pela leitura das classes, 
            #tenham sido ordenadas como 1 e 0, vai dar erro, se o que se espera é uma fuzzificação 
            #0=10 e 1=01, então para isso ordena-se antes
            arrayCategories = sorted(arrayCategories) 
            #for j in range(arraySets[i]): #necessário, caso as categorias sejam números
            #    arrayCategories[j] = str(df[nameCol[i]].unique()[j])
            j = 0 #posição no vetor de índices válidos
            for index in range(validNumberRows): #para cada linha válida
                for k in range(arraySets[i]): #para cada conjunto da variável atual
                    if arrayCategories[k] == df.loc[validIndexes[j]][i]: #quando as categorias são números, às vezes ocorrem problemas. Verificar se str() corrigiu
                        #se o objeto pertence à categoria atual, a posição na matriz de pertinência será 1
                        pertinenceMatrix[validIndexes[j],actualColumn+k] = 1
                    else:
                        #senão será 0
                        pertinenceMatrix[validIndexes[j],actualColumn+k] = 0
                j += 1
            actualColumn += arraySets[i] #todas as colunas referentes aos conjuntos da variável atual foram preenchidas
            actualIndexSets += 1
                
        else:# matrixDomain[i,0] != matrixDomain[i,1]: #dados reais
            lowerBound = matrixDomain[i,0] #início do domínio
            upperBound = matrixDomain[i,1] #fim do domínio
            width = (upperBound - lowerBound) / (arraySets[i] - 1) #largura do conjunto fuzzy, isto é, a largura da subida ou da descida
            step = (upperBound - lowerBound) / 1000
            
            #fuzzificação
            
            x = np.arange(lowerBound, upperBound + step, step)

            qual = [[[] for _ in range(validNumberRows)] for _ in range(arraySets[i])] #conjuntos fuzzy
            qual_level = [[] for _ in range(arraySets[i])] #valores de pertinência
    
            #primeiro termo fuzzy
            a = lowerBound - step
            b = lowerBound
            c = lowerBound + width
            qual[0] = fuzz.trimf(x, [a, b, c])
            
            #termos fuzzy do meio
            if arraySets[i] > 2:
                for j in range(arraySets[i]-2):#-1): #com o -1 vale para os do meio e o último
                    a = b
                    b = c
                    c = c + width
                    qual[j+1] = fuzz.trimf(x, [a, b, c])

            #último termo fuzzy
            a = upperBound - width
            b = upperBound
            c = upperBound + step
            qual[arraySets[i]-1] = fuzz.trimf(x, [a, b, c])
            
            m = 0
            for index in range(validNumberRows):
                data = DataFrame.loc[validIndexes[m]][i]
                #para evitar problemas com as extremidades
                if data <= lowerBound:
                    qual_level[0] = 1
                    pertinenceMatrix[validIndexes[m],actualColumn] = 1
                    for k in range(arraySets[i]-1):
                        qual_level[k+1] = 0
                        pertinenceMatrix[validIndexes[m],actualColumn+k+1] = 0
                elif data >= upperBound:
                    qual_level[arraySets[i]-1] = 1
                    pertinenceMatrix[validIndexes[m],actualColumn+arraySets[i]-1] = 1
                    for k in range(arraySets[i]-1):
                        qual_level[k] = 0
                        pertinenceMatrix[validIndexes[m],actualColumn+k] = 0
                else:
                    for k in range(arraySets[i]):
                        qual_level[k] = fuzz.interp_membership(x, qual[k], data)
                        pertinenceMatrix[validIndexes[m],actualColumn+k] = qual_level[k]
                m += 1
            actualColumn += arraySets[i]
            actualIndexSets += 1
            
    #cria dataframe a partir da matriz
    actualColumn = 0
    for i in range(numColumns):
        for j in range(arraySets[i]):
            pertinenceMatrixDF['{0}{1}{2}'.format(nameCol[i],'-',j)] = pertinenceMatrix[:,actualColumn+j]
        actualColumn += arraySets[i]
    pertinenceDataFrame = pd.DataFrame(pertinenceMatrixDF)      
    
    #elimina do dataframe final as mesmas linhas que foram removidas do dataframe inicial
    finalPertinenceDataFrame = pertinenceDataFrame.drop(totalIndexes)
            
    return finalPertinenceDataFrame