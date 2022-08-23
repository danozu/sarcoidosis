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

    filename = "sarcoidosis_experiment_3_normal_all"
    filename_dataset = "sn.csv"
    filename_crossval = "sn_cvi.mat"
    filename_matlab = "Exp_sarcoidose_sn.csv"
    filename_matlab_cc = "cc_" + filename_matlab
    filename_matlab_p = "p_" + filename_matlab
    crossval_field = 'SN_CrossValIndex'
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

    # Obtain the dataset
    df = pd.read_csv(filename_dataset)

    cols = list(df.columns)
    cols.pop()

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
        estimators.append(('SVM', SVC(probability=True, random_state=seed)))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'SVM__C': [1, 2, 5, 7, 10, 50, 100, 200, 400],
            'SVM__gamma': [0.001, 0.01, 0.05, 0.1, 1],
            'SVM__kernel': ['rbf']
        }

        options = {'Importance': True}

        svm_grid_result, svm_class, svm_auc, svm_probs, svm_preds, svm_score, svm_params, svm_features, svm_cc = evaluate_grid_model(
            df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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
            df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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
                df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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
                df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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
                df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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
        
        options = {'Importance': True}
        
        xgb_grid_result, xgb_class, xgb_auc, xgb_probs, xgb_preds, xgb_score, xgb_params, xgb_features, xgb_cc = evaluate_grid_model(
                df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

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

        options = {'Importance', 'DT'}

        dt_grid_result, dt_class, dt_auc, dt_probs, dt_preds, dt_score, dt_params, dt_features, dt_cc = evaluate_grid_model(
            df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, dt_class, [])
        y_probs = functools.reduce(operator.iconcat, dt_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        #====================
        # Logistic Regression
        #====================
        print('-- LOGISTIC REGRESSION MODEL --')
        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
                      'LOGR__C': [0.001, 0.01, 0.1, 1, 2, 3, 5, 10],
                      'LOGR__penalty': ['l2']  #

                      }

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('LOGR', LogisticRegression(random_state=seed)))
        model = Pipeline(estimators)

        options = {'Importance', 'LOGR'}

        log_grid_result, log_class, log_auc, log_probs, log_preds, log_score, log_params, log_features, log_cc = evaluate_grid_model(
            df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, log_class, [])
        y_probs = functools.reduce(operator.iconcat, log_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')
        
        # Store the result in dataframe
        
        class_df = pd.DataFrame(columns=['BFP', 'SVM', 'KNN', 'ADAB', 'RF', 'LGB', 'XGB', 'DT', 'LOGR'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['SVM'] = functools.reduce(operator.iconcat, svm_class, [])
        class_df['KNN'] = functools.reduce(operator.iconcat, knn_class, [])  # flatten the list of lists
        class_df['ADAB'] = functools.reduce(operator.iconcat, adab_class, [])
        class_df['RF'] = functools.reduce(operator.iconcat, rf_class, [])
        class_df['LGB'] = functools.reduce(operator.iconcat, lgb_class, [])
        class_df['XGB'] = functools.reduce(operator.iconcat, xgb_class, [])
        class_df['DT'] = functools.reduce(operator.iconcat, dt_class, [])
        class_df['LOGR'] = functools.reduce(operator.iconcat, log_class, [])
        class_df.to_csv(filename_class, index=False)

        params_df = pd.DataFrame(columns=['SVM', 'KNN', 'ADAB', 'RF', 'LGB', 'XGB', 'DT', 'LOGR'])
        params_df['SVM'] = pd.Series(svm_params)  # Transform to series
        params_df['KNN'] = pd.Series(knn_params)  # Transform to series
        params_df['ADAB'] = pd.Series(adab_params)  # Transform to series
        params_df['RF'] = pd.Series(rf_params)  # Transform to series
        params_df['LGB'] = pd.Series(lgb_params)  # Transform to series
        params_df['XGB'] = pd.Series(xgb_params)  # Transform to series
        params_df['DT'] = pd.Series(dt_params)  # Transform to series
        params_df['LOGR'] = pd.Series(log_params)  # Transform to series
        params_df.to_csv(filename_params, index=False)

        cc_df = pd.DataFrame(columns=['SVM', 'KNN', 'ADAB', 'RF', 'LGB', 'XGB', 'DT', 'LOGR'])
        cc_df['SVM'] = functools.reduce(operator.iconcat, svm_cc, [])  # Transform to series
        cc_df['KNN'] = functools.reduce(operator.iconcat, knn_cc, [])  # Transform to series
        cc_df['ADAB'] = functools.reduce(operator.iconcat, adab_cc, [])  # Transform to series
        cc_df['RF'] = functools.reduce(operator.iconcat, rf_cc, [])  # Transform to series
        cc_df['LGB'] = functools.reduce(operator.iconcat, lgb_cc, [])  # Transform to series
        cc_df['XGB'] = functools.reduce(operator.iconcat, xgb_cc, [])  # Transform to series
        cc_df['DT'] = functools.reduce(operator.iconcat, dt_cc, [])  # Transform to series
        cc_df['LOGR'] = functools.reduce(operator.iconcat, log_cc, [])  # Transform to series
        cc_df.to_csv(filename_cc, index=False)

        # Put the results together in dataframe
        result_df = pd.DataFrame(columns=['BFP', 'SVM', 'KNN', 'ADAB', 'RF', 'LGB', 'XGB', 'DT', 'LOGR', 'class'])
        result_df['BFP'] = bfp  # comes from matlab
        result_df['SVM'] = functools.reduce(operator.iconcat, svm_probs, []) # flatten the list of lists
        result_df['KNN'] = functools.reduce(operator.iconcat, knn_probs, []) # flatten the list of lists
        result_df['ADAB'] = functools.reduce(operator.iconcat, adab_probs, []) # flatten the list of lists
        result_df['RF'] = functools.reduce(operator.iconcat, rf_probs, []) # flatten the list of lists
        result_df['LGB'] = functools.reduce(operator.iconcat, lgb_probs, []) # flatten the list of lists
        result_df['XGB'] = functools.reduce(operator.iconcat, xgb_probs, []) # flatten the list of lists
        result_df['DT'] = functools.reduce(operator.iconcat, dt_probs, []) # flatten the list of lists
        result_df['LOGR'] = functools.reduce(operator.iconcat, log_probs, [])  # flatten the list of lists
        result_df['class'] = functools.reduce(operator.iconcat, knn_class, [])  # flatten the list of lists
        
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
