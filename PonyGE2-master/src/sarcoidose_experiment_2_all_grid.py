import time
#from scipy.io import loadmat
#import pandas as pd
#import numpy as np
#import functools  # flatten list of list
#import operator  # flatten List of list
from util import evaluate_grid_model, evaluate_bayes_model, cmp_class_results
from plot_roc import plotroc
from scipy.io import loadmat
import functools  # flatten list of list
import operator  # flatten List of list
#from matplotlib import pyplot
#from pandas import read_csv
#from pandas import set_option
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from skopt.space import Real, Categorical, Integer
from sklearn import metrics
#import matplotlib.pyplot as plt
#from sklearn.model_selection import StratifiedKFold
#from sklearn.linear_model import LogisticRegression

#from fuzzify import *
import pandas as pd
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

    # Obtain the datset
    df = pd.read_csv(filename_dataset)

    cols = list(df.columns)
    cols.pop()

    array = df.values

    print(df.describe())
    #
    df['class'].value_counts().plot(kind='bar', title='Count (class)')

    X = array[:, 0:16]
    Y = array[:, 16]

    # crossval_index = get_crossval_index(X, Y, 10, 1)

    models = [
        {
            'label': 'ponyGE',
            'model': 'ponyGE',
        }
    ]

    if Run == 1:
        seed = 7
        num_folds = 10
        num_class = 2
        n_iter = 30
        scoring = 'roc_auc'

        print("seed: %f" % (seed))
        print("n_iter: %d" % (n_iter))
        print("scoring: %s" % (scoring))

        #==========
        # PGLIQ_APF
        #==========
        print('-- ponyGE --')

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('ponyGE', ponyge(RANDOM_SEED=seed, MAX_INIT_TREE_DEPTH=4, MAX_TREE_DEPTH=5)))
        model = Pipeline(estimators)

        param_grid = {
            'ponyGE__GENERATIONS': [50, 100, 200],
            'ponyGE__POPULATION_SIZE': [100, 200, 500, 1000, 3000]
           # 'ponyGE__MAX_INIT_TREE_DEPTH': [4, 5],
           # 'ponyGE__MAX_TREE_DEPTH': [5, 6]


            
        }

        options = {'ponyge'}

        pgliq_apf_grid_result, pgliq_apf_class, pgliq_apf_auc, pgliq_apf_probs, pgliq_apf_preds, pgliq_apf_score, pgliq_apf_params, pgliq_apf_features, pgliq_apf_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, pgliq_apf_class, [])
        y_probs = functools.reduce(operator.iconcat, pgliq_apf_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Store the result in dataframe

        class_df = pd.DataFrame(columns=['BFP', 'ponyGE'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['ponyGE'] = functools.reduce(operator.iconcat, pgliq_apf_class, [])

        class_df.to_csv(filename_class, index=False)

        params_df = pd.DataFrame(columns=['ponyGE'])

        params_df['ponyGE'] = pd.Series(pgliq_apf_params)  # Transform to series

        params_df.to_csv(filename_params, index=False)

        cc_df = pd.DataFrame(columns=['ponyGE'])

        cc_df['ponyGE'] = functools.reduce(operator.iconcat, pgliq_apf_cc, [])  # Transform to series

        cc_df.to_csv(filename_cc, index=False)

        # Put the results together in dataframe
        result_df = pd.DataFrame(columns=['BFP', 'ponyGE', 'class'])
        result_df['BFP'] = bfp  # comes from matlab
        result_df['ponyGE'] = functools.reduce(operator.iconcat, pgliq_apf_probs, [])  # flatten the list of lists
        result_df['class'] = functools.reduce(operator.iconcat, pgliq_apf_class, [])  # flatten the list of lists
        
        result_df.to_csv(filename_result, index=False)

    else:
        print('Obtained results from files')
        result_df = pd.read_csv(filename_result)
        # auc_df = pd.read_csv(filename_auc)
        # cc_df = pd.read_csv(filename_cc)

    # Prepare to show the results

    roc_df, fig = plotroc(result_df, models, [0.90, 0.75])

    fig.savefig(filename_fig, format='png', dpi=300)

    roc_df.to_csv(filename_roc, index=True)

    print(roc_df)

    end = time.time()
    print("Elapsed time: ", end - start)
