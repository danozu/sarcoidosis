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

from gplearn.genetic import SymbolicClassifier
from gplearn.fitness import _Fitness
import numpy as np

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename = "sarcoidosis_experiment_2_altered_fuzzy_gp"
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

    # Obtain the dataset
    df = pd.read_csv(filename_dataset)

    cols = list(df.columns)
    cols.pop()

    models = [
        {
            'label': 'GP',
            'model': 'GP',
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
            'GP__init_depth': [(2, 2), (2, 6)],
            'GP__population_size': [100, 300, 500, 1000, 3000],
            'GP__tournament_size': [2, 7, 20],
            'GP__generations': [5, 20, 50, 100, 200],
            'GP__p_crossover': [0.8]
                      }

        # create pipeline
        estimators = []
        estimators.append(('standardize', StandardScaler()))
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

        options = {'GP', 'fuzzy'} 

        gp_grid_result, gp_class, gp_auc, gp_probs, gp_preds, gp_score, gp_params, gp_features, gp_cc = evaluate_grid_model(
            df, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, gp_class, [])
        y_probs = functools.reduce(operator.iconcat, gp_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Store the result in dataframe
        
        class_df = pd.DataFrame(columns=['BFP', 'GP'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['GP'] = functools.reduce(operator.iconcat, gp_class, [])

        class_df = pd.DataFrame(columns=['BFP'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['GP'] = functools.reduce(operator.iconcat, gp_class, [])

        class_df.to_csv(filename_class, index=False)

        params_df = pd.DataFrame(columns=['GP'])

        params_df['GP'] = pd.Series(gp_params)  # Transform to series

        params_df.to_csv(filename_params, index=False)

        cc_df = pd.DataFrame(columns=['GP'])

        cc_df['GP'] = functools.reduce(operator.iconcat, gp_cc, [])  # Transform to series

        cc_df.to_csv(filename_cc, index=False)

        # Put the results togpther in dataframe
        result_df = pd.DataFrame(columns=['BFP', 'GP', 'class'])
        result_df['BFP'] = bfp  # comes from matlab
        result_df['GP'] = functools.reduce(operator.iconcat, gp_probs, [])  # flatten the list of lists
        result_df['class'] = functools.reduce(operator.iconcat, gp_class, [])  # flatten the list of lists
        
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
