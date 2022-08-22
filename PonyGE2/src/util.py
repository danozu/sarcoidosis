import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.extmath import softmax
from sklearn.metrics.pairwise import pairwise_distances

from sklearn import metrics

import warnings

from joblib import Parallel, delayed

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


# def softmax (x):
#     B=np.exp(x)
#     C=np.sum(np.exp(x))
#     return B/C

def _parallel_execute(X,Y,crossval_index, model, i):
    """Private function used to build a batch of programs within a job."""
    print("Experimento %i" %i)
    train_index=crossval_index[i][0] # grab the train_index in expeiment i
    valid_index=crossval_index[i][1]
    test_index=crossval_index[i][2]
    train_index=train_index.flatten()
    valid_index=valid_index.flatten()
    test_index=test_index.flatten()
    train_index+=valid_index # it works since the class is 0 or 1
    X_train=X[train_index==1.0] # train dataset (features)
    Y_train=Y[train_index==1.0] # train dataset (class)
    X_test=X[test_index==1.0]
    Y_test=Y[test_index==1.0]
    
    clf = model.fit(X_train, Y_train, 5)
    prediction_test = model.predict_proba(X_test)
    preds = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction_test[:, 1], drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    score = auc
    print("%s: %f" %('auc', score))
    
    return prediction_test[:,1], preds, score, cmp_class_results(Y_test,preds), Y_test, auc

def _parallel_exe_val(X,Y,crossval_index, model, i):
    """Private function used to build a batch of programs within a job."""
    print("Experimento %i" %i)
    train_index=crossval_index[i][0] # grab the train_index in expeiment i
    valid_index=crossval_index[i][1]
    test_index=crossval_index[i][2]
    train_index=train_index.flatten()
    valid_index=valid_index.flatten()
    test_index=test_index.flatten()
    X_train=X[train_index==1.0] # train dataset (features)
    Y_train=Y[train_index==1.0] # train dataset (class)
    X_valid=X[valid_index==1.0] # train dataset (features)
    Y_valid=Y[valid_index==1.0] # train dataset (class)
    X_test=X[test_index==1.0]
    Y_test=Y[test_index==1.0]
    
    clf = model.fit_val(X_train, Y_train, X_valid, Y_valid)
    prediction_test = model.predict_proba(X_test)
    preds = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction_test[:, 1], drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    score = auc
    print("%s: %f" %('auc', score))
    
    return prediction_test[:,1], preds, score, cmp_class_results(Y_test,preds), Y_test, auc

def evaluate_pgliq(X,Y,crossval_index, model, options, num_class):
    probs_model=[]
    preds_model=[]
    params_model=[]
    score_model=[]
    features_model=[]
    cc_model=[]
    class_model=[]
    auc_model=[]
    experiments=np.shape(crossval_index)[0] #primeira dimensão de crossval_index
    print("Number of experiments: %d" % (experiments))
    
    results = Parallel(n_jobs=experiments)(
                delayed(_parallel_execute)(X,
                                           Y,
                                           crossval_index,
                                           model, 
                                           i)
                for i in range(experiments))
    
    for i in range(experiments):
        probs_model.append(results[i][0]) 
        preds_model.append(results[i][1]) 
        score_model.append(results[i][2]) 
        cc_model.append(results[i][3])  
        class_model.append(results[i][4]) 
        auc_model.append(results[i][5]) 
        
    grid_final = []
    return grid_final, class_model, auc_model, probs_model, preds_model, score_model, params_model, features_model, cc_model


def predict_proba(self, X):
    distances = pairwise_distances(X, self.centroids_, metric=self.metric)
    probs = softmax(-distances)
    return probs

# clf = NearestCentroid()
# clf.predict_proba = predict_proba.__get__(clf)
# clf.fit(X_train, y_train)
# clf.predict_proba(X_test)
def evaluate_model(X,Y,crossval_index, model, options, num_class):
    '''
        This is where the function's Document String (docstring) goes
        Evaluate the model with each partition.
        It does not perform parameter search
        '''
    probs_model=[]
    preds_model=[]
    params_model=[]
    score_model=[]
    features_model=[]
    cc_model=[]
    class_model=[]
    auc_model=[]
    experiments=np.shape(crossval_index)[0]
    print("Number of experiments: %d" % (experiments))
    for i in range(experiments):
        print("Fold: %d" % (i))
        train_index=crossval_index[i][0] # grab the train_index in expeiment i
        valid_index=crossval_index[i][1]
        test_index=crossval_index[i][2]
        train_index=train_index.flatten()
        valid_index=valid_index.flatten()
        test_index=test_index.flatten()
        train_index+=valid_index # it works since the class is 0 or 1
        X_train=X[train_index==1.0] # train dataset (features)
        Y_train=Y[train_index==1.0] # train dataset (class)
        X_test=X[test_index==1.0]
        Y_test=Y[test_index==1.0]


        model_result = model.fit(X_train, Y_train)
        model_result.score(X_test,Y_test)
        prediction_test=model_result.predict_proba(X_test) # calculate the probability
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction_test[:, 1], drop_intermediate=False)
        auc_model.append(metrics.auc(fpr, tpr))


        probs_model.append(prediction_test[:,1])
        preds_model.append(model_result.predict(X_test))
        score_model.append(model_result.score(X_test,Y_test))


        cc_model.append(cmp_class_results(Y_test,model_result.predict(X_test)))
        class_model.append(Y_test)

    model_final = model.fit(X, Y) # use only with other dataset
    return model_final, class_model, auc_model, probs_model, preds_model, score_model, cc_model

def evaluate_grid_model(X, Y, crossval_index, model, param_grid, scoring, num_folds, seed, options, num_class):
    probs_model = []
    preds_model = []
    params_model = []
    score_model = []
    features_model = []
    cc_model = []
    class_model = []
    auc_model = []
    experiments = np.shape(crossval_index)[0]  # primeira dimensão de crossval_index
    print("Number of experiments: %d" % (experiments))
    for i in range(experiments):
        train_index = crossval_index[i][0]  # grab the train_index in expeiment i
        valid_index = crossval_index[i][1]
        test_index = crossval_index[i][2]
        train_index = train_index.flatten()
        valid_index = valid_index.flatten()
        test_index = test_index.flatten()
        train_index += valid_index  # it works since the class is 0 or 1
        X_train = X[train_index == 1.0]  # train dataset (features)
        Y_train = Y[train_index == 1.0]  # train dataset (class)
        # Y_train = Y_train.reshape(-1,1)
        X_test = X[test_index == 1.0]
        Y_test = Y[test_index == 1.0]
        # Y_test =  Y_test.reshape(-1,1)
        # scaler = StandardScaler().fit(X_train)
        # rescaled_X_train = scaler.transform(X_train)
        # rescaled_X_test =  scaler.transform(X_test)
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        # kfold = KFold(n_splits=num_folds, random_state=seed)

        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, refit='True', n_jobs=16)
        grid_result = grid.fit(X_train, Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

        grid_result.score(X_test, Y_test)
        #'SymbolicRegressor' object has no attribute 'predict_proba'
        #Então calculo o predict, pois a interpretação que tenho das árvores fuzzy, é que o resultado já seja a probabilidade de ser da classe 1
        prediction_test = grid_result.predict_proba(X_test)  # calculate the probability
#        prediction = grid_result.predict(X_test)  # calculate the probability
#        prediction_test = np.vstack([1 - prediction, prediction]).T

        fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction_test[:, 1], drop_intermediate=False)
        auc_model.append(metrics.auc(fpr, tpr))

        # if 'Kbest' in options:
        #     features = grid_result.best_estimator_.named_steps['Kbest']
        #     res = [i for i, val in enumerate(features.get_support()) if val]
        #     features_model.append(res)
        #     print("Best Features: %s" %(str(res)[1:-1]))
        #
        if ('Importance' in options):
            features = grid_result.best_estimator_.named_steps['fs'].support_
            res = [i for i, x in enumerate(features) if x]
            features_model.append(res)
            print("Best Features: %s" % (str(res)[1:-1]))
            print("Features Importances: %s" % (str(grid_result.best_estimator_.named_steps['fs'].support_)[1:-1]))

        if ('Stability' in options):
            features = grid_result.best_estimator_.named_steps['fs'].get_support()
            res = [i for i, x in enumerate(features) if x]
            features_model.append(res)
            print("Best Features: %s" % (str(res)[1:-1]))
            print("Features Importances: %s" % (str(grid_result.best_estimator_.named_steps['fs'].get_support())[1:-1]))

        if ('GP' in options):
            print("Best individual: %s" %str(grid_result.best_estimator_.named_steps['GP']._program))
            print("Fitness: %s" % str(grid_result.best_estimator_.named_steps['GP']._program.raw_fitness_))
            print("Depth: %s" % str(grid_result.best_estimator_.named_steps['GP']._program.depth_))
            
        if ('LOGR' in options):
            print("Coefficient of the features in the decision function: ", grid_result.best_estimator_.named_steps['LOGR'].coef_)
            
        if ('Features' in options):
            features = grid_result.best_estimator_.named_steps['PGLIQ_APF'].features_
            media = np.mean(features)
            list_features = []
            list_features_index = []
            for i in range(len(features)):
                list_features.append(features[i])
                list_features_index.append([features[i], i])
            print("Final Probability of Features: %s" % (str(list_features)[1:-1]))
            ordened_list = sorted(list_features_index, reverse=True)
            best_features = []
            for i in range(len(features)):
                if ordened_list[i][0] >= media:
                    best_features.append(ordened_list[i][1])
            print("Best Features: %s" % (str(best_features)[1:-1]))

        if ('Built' in options):
            features = grid_result.best_estimator_.named_steps['PGLIQ_APF'].features_
            media = np.mean(features)
            list_features = []
            list_features_index = []
            for i in range(len(features)):
                list_features.append(features[i])
                list_features_index.append([features[i], i])
            print("Final Probability of Features: %s" % (str(list_features)[1:-1]))
            ordened_list = sorted(list_features_index, reverse=True)
            best_features = []
            for i in range(len(features)):
                if ordened_list[i][0] >= media:
                    best_features.append(ordened_list[i][1])
            print("Best Features: %s" % (str(best_features)[1:-1]))
            length = grid_result.best_estimator_.named_steps['PGLIQ_APF']._feature_length
            lisp = grid_result.best_estimator_.named_steps['PGLIQ_APF']._feature_Lisp
            print("Features Length: %s" % (str(length)[1:-1]))
            for i in range(len(lisp)):
                print("Feature %i: %s" %(i,lisp[i]))
        
        if ('ponyge' in options):
            individual = grid_result.best_estimator_.named_steps['ponyGE'].phenotype
            print("Final Individual: %s" % (str(individual)[0:-1]))
            
        if ('DT' in options):
            n_nodes = grid_result.best_estimator_.named_steps['DT'].tree_.node_count
            children_left = grid_result.best_estimator_.named_steps['DT'].tree_.children_left
            children_right = grid_result.best_estimator_.named_steps['DT'].tree_.children_right
            feature = grid_result.best_estimator_.named_steps['DT'].tree_.feature
            threshold = grid_result.best_estimator_.named_steps['DT'].tree_.threshold
            
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
            while len(stack) > 0:
                # `pop` ensures each node is only visited once
                node_id, depth = stack.pop()
                node_depth[node_id] = depth
            
                # If the left and right child of a node is not the same we have a split
                # node
                is_split_node = children_left[node_id] != children_right[node_id]
                # If a split node, append left and right children and depth to `stack`
                # so we can loop through them
                if is_split_node:
                    stack.append((children_left[node_id], depth + 1))
                    stack.append((children_right[node_id], depth + 1))
                else:
                    is_leaves[node_id] = True
            
            print(
                "The binary tree structure has {n} nodes and has "
                "the following tree structure:\n".format(n=n_nodes)
            )
            for i in range(n_nodes):
                if is_leaves[i]:
                    print(
                        "{space}node={node} is a leaf node.".format(
                            space=node_depth[i] * "\t", node=i
                        )
                    )
                else:
                    print(
                        "{space}node={node} is a split node: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                        )
                    )
            #tree.plot_tree(clf)
            #plt.show()

        probs_model.append(prediction_test[:, 1])
        preds_model.append(grid_result.predict(X_test))
        score_model.append(grid_result.best_score_)
        params_model.append(grid_result.best_params_)

        cc_model.append(cmp_class_results(Y_test, grid_result.predict(X_test)))
        class_model.append(Y_test)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=-1)
    # grid_final = grid.fit(X, Y) # use only with other dataset
    grid_final = []  # retirei essa parte, pois estava dando erro de over_sampling ao fazer o gridsearch na sampling_strategy do SMOTE()
    return grid_final, class_model, auc_model, probs_model, preds_model, score_model, params_model, features_model, cc_model

def evaluate_bayes_model(X,Y,crossval_index, model,param_grid,scoring,num_folds,n_iter,seed,options,num_class):
    probs_model=[]
    preds_model=[]
    params_model=[]
    score_model=[]
    features_model=[]
    cc_model=[]
    class_model=[]
    auc_model=[]
    experiments=np.shape(crossval_index)[0]
    print("Number of experiments: %d" % (experiments))
    for i in range(experiments):
        train_index=crossval_index[i][0] # grab the train_index in expeiment i
        valid_index=crossval_index[i][1]
        test_index=crossval_index[i][2]
        train_index=train_index.flatten()
        valid_index=valid_index.flatten()
        test_index=test_index.flatten()
        train_index+=valid_index # it works since the class is 0 or 1
        X_train=X[train_index==1.0] # train dataset (features)
        Y_train=Y[train_index==1.0] # train dataset (class)
        #Y_train = Y_train.reshape(-1,1)
        X_test=X[test_index==1.0]
        Y_test=Y[test_index==1.0]
        #Y_test =  Y_test.reshape(-1,1)
        #scaler = StandardScaler().fit(X_train)
        #rescaled_X_train = scaler.transform(X_train)
        #rescaled_X_test =  scaler.transform(X_test)
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        grid = BayesSearchCV(estimator=model, search_spaces=param_grid, scoring=scoring, cv=kfold, n_iter=n_iter,refit = 'True',n_jobs=-1)
        grid_result = grid.fit(X_train, Y_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        grid_result.score(X_test,Y_test)
        prediction_test=grid_result.predict_proba(X_test) # calculate the probability

        fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction_test[:, 1], drop_intermediate=False)
        auc_model.append(metrics.auc(fpr, tpr))

        # if 'Kbest' in options:
        #     features = grid_result.best_estimator_.named_steps['Kbest']
        #     res = [i for i, val in enumerate(features.get_support()) if val]
        #     features_model.append(res)
        #     print("Best Features: %s" %(str(res)[1:-1]))
        #
        if ('Importance' in options):
            features = grid_result.best_estimator_.named_steps['fs'].support_
            res = [i for i, x in enumerate(features) if x]
            features_model.append(res)
            print("Best Features: %s" %(str(res)[1:-1]))
            print("Features Importances: %s" % (str(grid_result.best_estimator_.named_steps['fs'].support_)[1:-1]))

        if ('Stability' in options):
            features = grid_result.best_estimator_.named_steps['fs'].get_support()
            res = [i for i, x in enumerate(features) if x]
            features_model.append(res)
            print("Best Features: %s" %(str(res)[1:-1]))
            print("Features Importances: %s" % (str(grid_result.best_estimator_.named_steps['fs'].get_support())[1:-1]))

        probs_model.append(prediction_test[:,1])
        preds_model.append(grid_result.predict(X_test))
        score_model.append(grid_result.best_score_)
        params_model.append(grid_result.best_params_)

        cc_model.append(cmp_class_results(Y_test,grid_result.predict(X_test)))
        class_model.append(Y_test)
    grid = BayesSearchCV(estimator=model, search_spaces=param_grid, scoring=scoring, cv=kfold)
    grid_final = grid.fit(X, Y) # use only with other dataset
    return grid_final, class_model, auc_model, probs_model, preds_model, score_model, params_model, features_model, cc_model
def cmp_class_results(a,b):
    cmp_cc=[]
    len_a = len(a)
    len_b = len(b)
    for i in range(len_a):
        if a[i] == b[i]:
            cmp_cc.append(1.0)
        else:
            cmp_cc.append(0.0)
    return cmp_cc



print("End of Program")