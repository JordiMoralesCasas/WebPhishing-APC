import sys
sys.path.insert(1, 'helpers')

from imports import *
from pca import *


np.random.seed(1)
torch.manual_seed(1)


def load_dataset(file):
    """
    Returns a datasest given a file.
            Parameters:
                    file (string): csv file that contains de data.

            Returns:
                    x (pandas dataframe): Dataset with the independent variables
                    y (pandas dataframe): Dataset with the objective variable
    """

    ds_original = pd.read_csv(file, header=0, delimiter=',',decimal=',')

    # The objective is to predict the status of an url (Objective variable)
    y = ds_original[:]["status"] 
    x = ds_original.drop(columns = "status")

    return x, y


def object_type_to_float(x):
    """
    Convert the unknown "object" type data in the dataset to float.
            Parameters:
                    x (pandas dataframe): Dataset

            Returns:
                    x (pandas dataframe): Converted dataset
    """
    
    for col in range(x.shape[1]):
        if x.dtypes[col] == object:
            x[x.columns[col]] = x[x.columns[col]].astype('float')

    return x


def standardize(x, categorical_features = []):
    """
    Standardizes de data from a dataset.
            Parameters:
                    x (pandas dataframe): Dataset
                    categorical_features (string list): List with the categorical features that shouldn't be processed

            Returns:
                    x_standard (pandas dataframe): Dataset with standardized data
    """
    non_categorical_features = [i for i in x.columns if (i not in categorical_features)]

    # Subsets of x for categorical and non-categorical features
    x_non_categorical = x[non_categorical_features]
    x_categorical = x[categorical_features]

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler()
    x_non_categorical_standard = scaler.fit_transform(x_non_categorical)

    # Merge all the data into a single dataframe 
    df = pd.DataFrame(x_non_categorical_standard)
    df.columns = non_categorical_features
    x_standard = pd.concat([df, x_categorical], axis = 1) # Standardized data
    
    return x_standard


def split_data(x, y, train_ratio=0.8):
    """
    Splits the data into two subsets (train and test). The size of each subset is given by the provided ratio.
            Parameters:
                    x (pandas dataframe): Dataset with the independent variables
                    y (pandas dataframe): Dataset with the objective variable
                    train_ratio (float): Size of the train subset with respecto to the initial dataset
            Returns:
                    x_train (numpy array): Train subset with the independent variables
                    y_train (numpy array): Train subset with the objective variable
                    x_test (numpy array): Test subset with the independent variables
                    y_test (numpy array): Test subset with the objective variable
    """
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))

    indices_train, indices_test = indices[:n_train], indices[n_train:]

    x_train, y_train = x[indices_train, :], y[indices_train]
    x_test, y_test = x[indices_test, :], y[indices_test]

    return x_train, y_train, x_test, y_test


# inspired by https://machinelearningmastery.com/feature-selection-with-categorical-data/
def select_features(x, y, k):
    """
    Selects the k best features of a dataset.
            Parameters:
                    x (pandas dataframe): Dataset with the independent variables
                    y (pandas dataframe): Dataset with the objective variable
                    k (int): Number of features to select
            Returns:
                    indices (int list): Indices of the best k features

    """
    # Split the data
    x_train, y_train, x_test, y_test = split_data(x.values, y)

    # Perform a feature selection based on the mutual information between each independent
    # variable and the objective variable.
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(x_train, y_train)

    X_train_fs, X_test_fs = fs.transform(x_train), fs.transform(x_test)

    # K best features
    indices = fs.scores_.argsort()[::-1]

    return indices[:k]