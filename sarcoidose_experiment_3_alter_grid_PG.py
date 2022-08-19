import time
# from scipy.io import loadmat
import pandas as pd
import numpy as np
# import functools  # flatten list of list
# import operator  # flatten List of list
from util import evaluate_grid_model, evaluate_bayes_model, cmp_class_results
from plot_roc import plotroc
from scipy.io import loadmat
import functools  # flatten list of list
import operator  # flatten List of list
# from matplotlib import pyplot
# from pandas import read_csv
# from pandas import set_option
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
# from skopt.space import Real, Categorical, Integer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

from gplearn.genetic import SymbolicClassifier
from gplearn.fitness import _Fitness

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename = "sarcoidose_experiment_3_alter"
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
            'label': 'GP',
            'model': 'GP',
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

        # ==========
        # PGLIQ_APF
        # ==========
        print('-- GP --')

        def _roc_auc(y, y_pred, w):
            """Calculate the AUC."""
            proba = np.vstack([1 - y_pred, y_pred]).T
            fpr, tpr, threshold = roc_curve(y, proba[:, 1])
            return auc(fpr, tpr)

        #roc_auc = make_fitness(_roc_auc, greater_is_better=True)
        roc_auc = _Fitness(function=_roc_auc, greater_is_better=True)
        
        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'GP__init_depth': [(2, 3), (2, 10)],
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
                                                    stopping_criteria=0.01, n_jobs=5,
                                                    p_subtree_mutation=0.01, 
                                                    p_point_replace=0.05, 
                                                    #init_depth=(2,10),
                                                    function_set=('add', 'sub', 'mul', 'div'),
                                                    init_method='half and half',
                                                    const_range=(-1.,1.),
                                                    p_hoist_mutation=0.01, #combate bloat 0.05
                                                    p_point_mutation=0.01,
                                                    #tournament_size=20,
                                                    max_samples=1, verbose=0,
                                                    parsimony_coefficient=0.001, low_memory=True)))

        model = Pipeline(estimators)

        options = {'Importance', 'GP'}  # {'ponyge'}

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