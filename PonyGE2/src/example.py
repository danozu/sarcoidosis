#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:01:08 2020

@author: allan
"""

import time
from scipy.io import loadmat
import pandas as pd
import numpy as np
import functools  # flatten list of list
import operator  # flatten List of list
from util import evaluate_grid_model, evaluate_pgliq#, cmp_class_results
from plot_roc import plotroc
from scipy.io import loadmat
import functools  # flatten list of list
import operator  # flatten List of list
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from fuzzify import *

from genetic_ponyge import *

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename = "sarcoidose_experiment_2_all"
    filename_dataset = "sr.csv"
    filename_crossval = "sr_cvi.mat"
    filename_matlab = "Exp_sarcoidose_sr.csv"
    filename_matlab_cc = "cc_" + filename_matlab
    filename_matlab_p = "p_" + filename_matlab
    crossval_field = 'SR_CrossValIndex'
    filename_csv = filename + ".csv"
    filename_roc = "roc_" + filename_csv
    filename_auc = "auc_" + filename_csv
    filename_class = "class_" + filename_csv
    filename_result = "result_" + filename_csv
    filename_cc = "cc_" + filename_csv
    filename_params = "params_" + filename_csv
    filename_fig = filename + ".png"

    cell_mat = loadmat(filename_crossval)  # load crossval index
    print(cell_mat.keys())
    crossval_index = cell_mat[crossval_field]
    print(type(crossval_index))
    print(crossval_index.shape[0])
    train_index = crossval_index[0][0]
    valid_index = crossval_index[0][1]
    test_index = crossval_index[0][2]
    train_index = train_index.flatten()
    valid_index = valid_index.flatten()
    test_index = test_index.flatten()
    print(train_index)
    print(valid_index)
    print(test_index)

    # Obtain the result for best FOT parameter from matlab

    matlab_df = pd.read_csv(filename_matlab, delimiter=',', encoding="utf-8-sig")
    #
    print(matlab_df.head())
    print(matlab_df.columns)
    # array = bfp_df.values
    bfp = matlab_df['BFP']

    # Copy the cc values from Matlab
    matlab_cc_df = pd.read_csv(filename_matlab_cc, delimiter=',', encoding="utf-8-sig")
    bfp_cc = matlab_cc_df['BFP']

    #Obtain the datset
    df = pd.read_csv(filename_dataset)

    #Fuzzificando o dataframe
#    dominio = matrixDomain(df)
#    datapd = fuzzifyDataFrame(df,3,dominio) #17*3 colunas
#    _, cfuzzificadas = np.shape(datapd) #cfuzzificadas=50

#    cols = list(datapd.columns)
    cols = list(df.columns)
#    cols.pop()
    cols.pop()

#    array = datapd.values
    array = df.values

    print(df.describe())
#    print(datapd.describe())

    #
    df['class'].value_counts().plot(kind='bar', title='Count (class)')

#    X = array[:, 0:(cfuzzificadas-2)]
#    Y = array[:, (cfuzzificadas-1)]
    X = array[:, 0:16]
    Y = array[:, 16]

    models = [
        {
            'label': 'C_PGLIQ_APF_0',
            'model': 'C_PGLIQ_APF_0',
        },
        {
            'label': 'C_PGLIQ_APF_1',
            'model': 'C_PGLIQ_APF_1',
        },
        {
            'label': 'C_PGLIQ_APF_2',
            'model': 'C_PGLIQ_APF_2',
        }
    ]

    if Run == 1:
        seed = 7
        num_class = 2

        print("seed: %f" % (seed))

        model_fitness = LogisticRegression(C=1, penalty='l2', max_iter=100)

        print('-- C_PGLIQ_APF_0 --')
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('PGLIQ_APF', FPTBuilder(model_fitness=model_fitness,n_new_features=2,generations=10000,individual_length=4,metric='roc_auc',feature_names=cols,random_state=seed)))
        model = Pipeline(estimators)
        #model = FPTBuilder(model_fitness=model_fitness,n_new_features=2,generations=10000,individual_length=4,metric='roc_auc',feature_names=cols,random_state=seed)
        param_grid = {}
        options = {}
        pgliq_apf_grid_result, pgliq_apf_class, pgliq_apf_auc, pgliq_apf_probs, pgliq_apf_preds, pgliq_apf_score, pgliq_apf_params, pgliq_apf_features, pgliq_apf_cc = evaluate_pgliq(
            X, Y, crossval_index, model, options, num_class)

        classes = functools.reduce(operator.iconcat, pgliq_apf_class, [])
        y_probs = functools.reduce(operator.iconcat, pgliq_apf_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        print('-- C_PGLIQ_APF_1 --')
        model = FPTBuilder(model_fitness=model_fitness,n_new_features=2,generations=10000,individual_length=8,metric='roc_auc',feature_names=cols,random_state=seed)
        param_grid = {}
        options = {}
        pgliq_apf_grid_result2, pgliq_apf_class2, pgliq_apf_auc2, pgliq_apf_probs2, pgliq_apf_preds2, pgliq_apf_score2, pgliq_apf_params2, pgliq_apf_features2, pgliq_apf_cc2 = evaluate_pgliq(
            X, Y, crossval_index, model, options, num_class)

        classes = functools.reduce(operator.iconcat, pgliq_apf_class2, [])
        y_probs = functools.reduce(operator.iconcat, pgliq_apf_probs2, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        print('-- C_PGLIQ_APF_2 --')
        model = FPTBuilder(model_fitness=model_fitness,n_new_features=2,generations=10000,individual_length=16,metric='roc_auc',feature_names=cols,random_state=seed)
        param_grid = {}
        options = {}
        pgliq_apf_grid_result3, pgliq_apf_class3, pgliq_apf_auc3, pgliq_apf_probs3, pgliq_apf_preds3, pgliq_apf_score3, pgliq_apf_params3, pgliq_apf_features3, pgliq_apf_cc3 = evaluate_pgliq(
            X, Y, crossval_index, model, options, num_class)

        classes = functools.reduce(operator.iconcat, pgliq_apf_class3, [])
        y_probs = functools.reduce(operator.iconcat, pgliq_apf_probs3, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Store the result in dataframe

        # Check if the class are the same

        class_df = pd.DataFrame(columns=['BFP', 'C_PGLIQ_APF_0', 'C_PGLIQ_APF_1', 'C_PGLIQ_APF_2'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['C_PGLIQ_APF_0'] = functools.reduce(operator.iconcat, pgliq_apf_class, [])
        class_df['C_PGLIQ_APF_1'] = functools.reduce(operator.iconcat, pgliq_apf_class2, [])
        class_df['C_PGLIQ_APF_2'] = functools.reduce(operator.iconcat, pgliq_apf_class3, [])

        class_df.to_csv(filename_class, index=False)

        params_df = pd.DataFrame(columns=['C_PGLIQ_APF_0', 'C_PGLIQ_APF_1', 'C_PGLIQ_APF_2'])

        params_df['C_PGLIQ_APF_0'] = pd.Series(pgliq_apf_params)  # Transform to series
        params_df['C_PGLIQ_APF_1'] = pd.Series(pgliq_apf_params2)  # Transform to series
        params_df['C_PGLIQ_APF_2'] = pd.Series(pgliq_apf_params3)  # Transform to series

        params_df.to_csv(filename_params, index=False)

        cc_df = pd.DataFrame(columns=['C_PGLIQ_APF_0', 'C_PGLIQ_APF_1', 'C_PGLIQ_APF_2'])

        cc_df['C_PGLIQ_APF_0'] = functools.reduce(operator.iconcat, pgliq_apf_cc, [])  # Transform to series
        cc_df['C_PGLIQ_APF_1'] = functools.reduce(operator.iconcat, pgliq_apf_cc2, [])  # Transform to series
        cc_df['C_PGLIQ_APF_2'] = functools.reduce(operator.iconcat, pgliq_apf_cc3, [])  # Transform to series

        cc_df.to_csv(filename_cc, index=False)

        # Put the results together in dataframe
        result_df = pd.DataFrame(columns=['BFP', 'C_PGLIQ_APF_0', 'C_PGLIQ_APF_1', 'C_PGLIQ_APF_2', 'class'])
        result_df['BFP'] = bfp  # comes from matlab
        result_df['C_PGLIQ_APF_0'] = functools.reduce(operator.iconcat, pgliq_apf_probs, [])  # flatten the list of lists
        result_df['C_PGLIQ_APF_1'] = functools.reduce(operator.iconcat, pgliq_apf_probs2, [])  # flatten the list of lists
        result_df['C_PGLIQ_APF_2'] = functools.reduce(operator.iconcat, pgliq_apf_probs3, [])  # flatten the list of lists
        result_df['class'] = functools.reduce(operator.iconcat, pgliq_apf_class, [])  # flatten the list of lists

        result_df.to_csv(filename_result, index=False)

    else:
        print('Obtained results from files')
        result_df = pd.read_csv(filename_result)

    # Prepare to show the results

    roc_df, fig = plotroc(result_df, models, [0.90, 0.75])

    fig.savefig(filename_fig, format='png', dpi=300)

    roc_df.to_csv(filename_roc, index=True)

    print(roc_df)

    end = time.time()
    print("Elapsed time: ", end - start)


