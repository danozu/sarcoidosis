import time
import pandas as pd
from util import evaluate_grid_model
from plot_roc import plotroc
from scipy.io import loadmat
import functools  # flatten list of list
import operator  # flatten List of list

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

from genetic_ponyge import ponyge

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename = "sarcoidosis_experiment_3_altered_ge"
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
            'label': 'GE',
            'model': 'GE',
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
        
        #======================
        # Grammatical Evolution
        #======================
        print('-- GE --')

        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('fs', feat_selection))
        estimators.append(('GE', ponyge(CROSSOVER_PROBABILITY=0.8, MUTATION_PROBABILITY=0.01, 
                                            MAX_TREE_DEPTH=17, RANDOM_SEED=seed)))
        model = Pipeline(estimators)

        param_grid = {
            'fs__n_features_to_select': [4, 8, 12],
            'GE__MAX_INIT_TREE_DEPTH': [4, 12],
            'GE__TOURNAMENT_SIZE': [2, 7, 20],
            'GE__GENERATIONS': [5, 20, 50, 100, 200],
            'GE__POPULATION_SIZE': [100, 300, 500, 1000, 3000],
        }

        options = {'ponyge'}

        ge_grid_result, ge_class, ge_auc, ge_probs, ge_preds, ge_score, ge_params, ge_features, ge_cc = evaluate_grid_model(
            X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class)

        classes = functools.reduce(operator.iconcat, ge_class, [])
        y_probs = functools.reduce(operator.iconcat, ge_probs, [])  # flatten the list of lists
        fpr, tpr, thresholds = metrics.roc_curve(classes, y_probs, drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')

        # Store the result in dataframe
        
        class_df = pd.DataFrame(columns=['BFP', 'GE'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['GE'] = functools.reduce(operator.iconcat, ge_class, [])

        class_df = pd.DataFrame(columns=['BFP'])
        class_df['BFP'] = matlab_df['class']  # comes from matlab
        class_df['GE'] = functools.reduce(operator.iconcat, ge_class, [])

        class_df.to_csv(filename_class, index=False)

        params_df = pd.DataFrame(columns=['GE'])

        params_df['GE'] = pd.Series(ge_params)  # Transform to series

        params_df.to_csv(filename_params, index=False)

        cc_df = pd.DataFrame(columns=['GE'])

        cc_df['GE'] = functools.reduce(operator.iconcat, ge_cc, [])  # Transform to series

        cc_df.to_csv(filename_cc, index=False)

        # Put the results together in dataframe
        result_df = pd.DataFrame(columns=['BFP', 'GE', 'class'])
        result_df['BFP'] = bfp  # comes from matlab
        result_df['GE'] = functools.reduce(operator.iconcat, ge_probs, [])  # flatten the list of lists
        result_df['class'] = functools.reduce(operator.iconcat, ge_class, [])  # flatten the list of lists
        
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
