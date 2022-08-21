import time
import pandas as pd
from util import evaluate_grid_model
from plot_roc import plotroc
from scipy.io import loadmat
import functools  # flatten list of list
import operator  # flatten List of list

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn import tree

from PonyGE2.src.genetic_ponyge import ponyge

from gplearn.genetic import SymbolicClassifier
from gplearn.fitness import _Fitness

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename = "sarcoidosis_experiment_3_altered"
    filename_dataset = "sa.csv"
    filename_crossval = "sa_cvi.mat"
    filename_matlab = "Exp_sarcoidose_sa.csv"
    filename_matlab_cc = "cc_" + filename_matlab
    filename_matlab_p = "p_" + filename_matlab
    crossval_field = 'SA_CrossValIndex'
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
    df['class'].value_counts().plot(kind='bar', title='Count (class)')

    X = array[:, 0:16]
    Y = array[:, 16]

    models = [
        {
            'label': 'SVM',
            'model': 'SVM',

        },
        {
            'label': 'KNN',
            'model': 'KNN',

        },
        {
            'label': 'ADAB',
            'model': 'ADAB',

        },
        {
            'label': 'RF',
            'model': 'RF',
        },
        {
            'label': 'LGB',
            'model': 'LGB',

        },
        {
            'label': 'XGB',
            'model': 'XGB',

        },
        {
            'label': 'DT',
            'model': 'DT',

        },
        {
            'label': 'LOGR',
            'model': 'LOGR',

        },
        {
            'label': 'GP',
            'model': 'GP',

        },
        {
            'label': 'ponyGE',
            'model': 'ponyGE',
        }
    ]
    
    mfs = LinearSVC(C=1, penalty="l1", dual=False, max_iter=5000)

    feat_selection = RFE(mfs, step=1)

    if Run == 1:
        seed = 7
        num_folds = 10
        num_class = 2
        n_iter = 30
        scoring = 'roc_auc'

        print("seed: %f" % (seed))
        print("n_iter: %d" % (n_iter))
        print("scoring: %s" % (scoring))
        
        # SVM
        print('-- SVM MODEL --')

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('SVM', SVC(probability=True)))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'SVM__C': [1, 2, 5, 7, 10, 50, 100, 200, 400],
            'SVM__gamma': [0.001, 0.01, 0.05, 0.1, 1],
            'SVM__kernel': ['rbf']
        }

        options = {'Importance': True}

        svm_grid_result, svm_class, svm_auc, svm_probs, svm_preds, svm_score, svm_params, svm_features, svm_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, svm_class, [])
        y_probs = functools.reduce(operator.iconcat, svm_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # KNN
        print('-- KNN MODEL --')

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('KNN', KNeighborsClassifier(metric='manhattan')))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'KNN__n_neighbors': [1, 3, 5, 7, 9, 11, 13],
            'KNN__weights': ['distance']
        }

        options = {'Importance': True}

        knn_grid_result, knn_class, knn_auc, knn_probs, knn_preds, knn_score, knn_params, knn_features, knn_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, knn_class, [])
        y_probs = functools.reduce(operator.iconcat, knn_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Random Forest
        print('-- RF MODEL --')

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'RF__n_estimators': [10, 30, 60, 100, 200, 400],
            'RF__max_depth': [1, 2, 3, 4, 5, 10, 15, 30, 60]
        }

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('RF', RandomForestClassifier(random_state=seed)))
        model = Pipeline(estimators)

        options = {'Importance': True}

        rf_grid_result, rf_class, rf_auc, rf_probs, rf_preds, rf_score, rf_params, rf_features, rf_cc = evaluate_grid_model(
                X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, rf_class, [])
        y_probs = functools.reduce(operator.iconcat, rf_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Adaboost
        print('-- ADAB MODEL --')

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'ADAB__base_estimator__max_depth': [1, 2, 3, 4, 5, 10, 15, 30, 60],
            'ADAB__n_estimators': [10, 30, 60, 100, 200, 400]
        }

        DTC = DecisionTreeClassifier(max_features="auto", class_weight="balanced", max_depth=None, random_state=seed)

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('ADAB', AdaBoostClassifier(base_estimator=DTC, random_state=seed)))
        model = Pipeline(estimators)

        options = {'Importance': True}

        adab_grid_result, adab_class, adab_auc, adab_probs, adab_preds, adab_score, adab_params, adab_features, adab_cc = evaluate_grid_model(
                X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, adab_class, [])
        y_probs = functools.reduce(operator.iconcat, adab_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Lightgbm
        print('-- LGB MODEL --')

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'LGB__max_depth': [1, 2, 3, 4, 5, 10, 15, 30, 60],
            'LGB__n_estimators': [10, 30, 60, 100, 200, 400]
        }

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('LGB', LGBMClassifier(random_state=seed, max_depth=0)))
        model = Pipeline(estimators)

        options = {'Importance': True}

        lgb_grid_result, lgb_class, lgb_auc, lgb_probs, lgb_preds, lgb_score, lgb_params, lgb_features, lgb_cc = evaluate_grid_model(
                X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, lgb_class, [])
        y_probs = functools.reduce(operator.iconcat, lgb_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        
        print('-- XGB MODEL --')

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'XGB__max_depth': [1, 2, 3, 4, 5, 10, 15, 30, 60],
            'XGB__n_estimators': [10, 30, 60, 100, 200, 400]
        }

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('XGB', XGBClassifier(random_state=seed)))
        model = Pipeline(estimators)
        xgb_grid_result, xgb_class, xgb_auc, xgb_probs, xgb_preds, xgb_score, xgb_params, xgb_features, xgb_cc = evaluate_grid_model(
                X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, xgb_class, [])
        y_probs = functools.reduce(operator.iconcat, xgb_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')
        
        #===============
        # Decision Trees
        #===============
        print('-- DT MODEL --')

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('DT', tree.DecisionTreeClassifier(random_state=seed)))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'DT__max_depth': [2, 3, 4, 5, 10, 50],
            'DT__criterion': ['gini', 'entropy', 'log_loss'],
            'DT__splitter': ['best', 'random']
        }

        options = {'DT'}

        dt_grid_result, dt_class, dt_auc, dt_probs, dt_preds, dt_score, dt_params, dt_features, dt_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, dt_class, [])
        y_probs = functools.reduce(operator.iconcat, dt_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        #====================
        # Logistic Regression
        #====================
        print('-- LOGISTIC REGRESSION MODEL --')
        param_grid = {'fs__n_features_to_select': [4, 8, 12],
                      'LOGR__C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10],
                      'LOGR__penalty': ['l2']  #

                      }

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('LOGR', LogisticRegression(random_state=seed)))
        model = Pipeline(estimators)

        options = {'LOGR'}

        log_grid_result, log_class, log_auc, log_probs, log_preds, log_score, log_params, log_features, log_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, log_class, [])
        y_probs = functools.reduce(operator.iconcat, log_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')
        
        # ===================
        # Genetic Programming
        # ===================
        print('-- GP --')

        def _roc_auc(y, y_pred, w):
            """Calculate the AUC."""
            proba = np.vstack([1 - y_pred, y_pred]).T
            fpr, tpr, threshold = metrics.roc_curve(y, proba[:, 1])
            return metrics.auc(fpr, tpr)

        roc_auc = _Fitness(function=_roc_auc, greater_is_better=True)
        
        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'GP__init_depth': [(2, 2), (2, 6)],
            'GP__population_size': [100, 300, 500, 1000, 3000],
            'GP__tournament_size': [2, 7, 20],
            'GP__generations': [5, 20, 50, 100, 200],
            'GP__p_crossover': [0.8]
                      }

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('GP', SymbolicClassifier(random_state=seed, #feature_names=cols,
                                                    metric=roc_auc,
                                                    stopping_criteria=0.01, n_jobs=16,
                                                    p_subtree_mutation=0.01, 
                                                    p_point_replace=0.05, 
                                                    function_set=('add', 'sub', 'mul', 'div'),
                                                    init_method='half and half',
                                                    const_range=(-1.,1.),
                                                    p_hoist_mutation=0.01, #combate bloat 0.05
                                                    p_point_mutation=0.01,
                                                    max_samples=1, verbose=0,
                                                    parsimony_coefficient=0.001, low_memory=True)))

        model = Pipeline(estimators)

        options = {'Importance', 'GP'} 

        pgliq_apf_grid_result, pgliq_apf_class, pgliq_apf_auc, pgliq_apf_probs, pgliq_apf_preds, pgliq_apf_score, pgliq_apf_params, pgliq_apf_features, pgliq_apf_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, pgliq_apf_class, [])
        y_probs = functools.reduce(operator.iconcat, pgliq_apf_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        #======================
        # Grammatical Evolution
        #======================
        print('-- ponyGE --')

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('ponyGE', ponyge(CROSSOVER_PROBABILITY=0.8, MUTATION_PROBABILITY=0.01, 
                                            MAX_TREE_DEPTH=17, RANDOM_SEED=seed)))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'ponyGE__MAX_INIT_TREE_DEPTH': [4, 12],
            'ponyGE__TOURNAMENT_SIZE': [2, 7, 20],
            'ponyGE__GENERATIONS': [5, 20, 50, 100, 200],
            'ponyGE__POPULATION_SIZE': [100, 300, 500, 1000, 3000],

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
