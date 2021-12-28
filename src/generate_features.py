import sys
sys.path.insert(1, 'helpers')

from imports import *
from pca import *


np.random.seed(1)
torch.manual_seed(1)


def load_dataset(file):
    ds_original = pd.read_csv(file, header=0, delimiter=',',decimal=',')


    # The objective is to predict the status of an url (Dependent variable)
    y = ds_original[:]["status"] 
    x = ds_original.drop(columns = "status")

    return x, y


def data_to_float(x):
    # The x columns with float numbers are actually stored as an unknown _object_ variable. A conversion to float
    # type is needed
    type_of_cols = [x.dtypes == object][0]
    for col in range(x.shape[1]):
        if type_of_cols[col]:
            x[x.columns[col]] = x[x.columns[col]].astype('float')

    return x




def standardize(x, categorical_features = []):
    non_categorical_features = [i for i in x.columns if (i not in categorical_features)]

    x_non_categorical = x[non_categorical_features]
    x_categorical = x[categorical_features]

    scaler = StandardScaler() # Standardize features by removing the mean and scaling to unit variance.
    #scaler = MinMaxScaler()

    x_non_categorical_standard = scaler.fit_transform(x_non_categorical)

    df = pd.DataFrame(x_non_categorical_standard)
    df.columns = non_categorical_features

    x_standard = pd.concat([df, x_categorical], axis = 1) # Standardized data
    
    return x_standard

# For each atribute we will perform a linear regression so we can compute its r2 score. We will select the ones
# with a higher score
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val

def selection_reg(x_standard, y):
    x_train, y_train, x_val, y_val = split_data(x_standard.values, y)

    r2_table = np.zeros((x_train.shape[1], 2))

    # Linear regression for every attribute
    for i in range(x_train.shape[1]):
        x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
        x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))

        regr = LinearRegression()
        regr.fit(x_t, y_train)
        
        r2 = r2_score(y_val, regr.predict(x_v))
        
        r2_table[i, 1] = r2
        r2_table[i, 0] = i

    # Select the best atributes (15 atributes)
    best_atributes = r2_table[r2_table[:, 1].argsort()[::-1]][:,0].astype('int')
    return best_atributes


# inspired by https://machinelearningmastery.com/feature-selection-with-categorical-data/
def select_features(x, y, k):
    # Split the data
    x_train, y_train, x_val, y_val = split_data(x.values, y)


    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(x_train, y_train)
    X_train_fs = fs.transform(x_train)
    X_test_fs = fs.transform(x_val)

    indices = fs.scores_.argsort()[::-1]

    return indices[:k]