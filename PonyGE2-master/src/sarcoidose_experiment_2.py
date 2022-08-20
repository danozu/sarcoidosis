import time
from scipy.io import loadmat
from sklearn.pipeline import Pipeline

from genetic_ponyge import *
from fuzzify import *

if __name__ == '__main__':
    start = time.time()

    Run = 1

    filename_dataset = "sr.csv"
    filename_crossval = "sr_cvi.mat"
    crossval_field = 'SR_CrossValIndex'

    cell_mat = loadmat(filename_crossval)  # load crossval index
    print(cell_mat.keys())
    crossval_index = cell_mat[crossval_field]
    print(type(crossval_index))
    print(crossval_index.shape[0])
    i=0
    train_index=crossval_index[i][0] # grab the train_index in expeiment i
    valid_index=crossval_index[i][1]
    test_index=crossval_index[i][2]
    train_index=train_index.flatten()
    valid_index=valid_index.flatten()
    test_index=test_index.flatten()
    train_index+=valid_index # it works since the class is 0 or 1
    print(train_index)
    print(valid_index)
    print(test_index)

    #Obtain the datset
    df = pd.read_csv(filename_dataset)
    
    #Fuzzificando o dataframe
    dominio = matrixDomain(df)
    datapd = fuzzifyDataFrame(df,3,dominio) #17*3 colunas
    _, cfuzzificadas = np.shape(datapd) #cfuzzificadas=50

    cols = list(datapd.columns)
    cols.pop()
    cols.pop()
    
    array = datapd.values
    
    print(datapd.describe())

    #
    df['class'].value_counts().plot(kind='bar', title='Count (class)')

    X = array[:, 0:(cfuzzificadas-2)]
    Y = array[:, (cfuzzificadas-1)]
    
    X_train=X[train_index==1.0] # train dataset (features)
    Y_train=Y[train_index==1.0] # train dataset (class)
    X_test=X[test_index==1.0]
    Y_test=Y[test_index==1.0]
    
    estimators = []
    estimators.append(('ponyge', ponyge(GENERATIONS = 100, POPULATION_SIZE=100)))
    model = Pipeline(estimators)
    #model = ponyge()
    
    clf = model.fit(X_train,Y_train)
    prediction_test = model.predict_proba(X_test)
    preds = model.predict(X_test)

    end = time.time()
    print("Elapsed time: ", end - start)