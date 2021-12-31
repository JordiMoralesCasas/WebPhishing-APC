import sys
sys.path.insert(1, 'helpers')

from imports import *
from generate_features import *
from utils import *
from pca import *
from train_models import *

def score_logistic_torch(file, dataset):
    """
    Train and score a PyTorch Logistic Regression model, given a model file.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset
                    show_confusion_matrix (bool): If True, a confusion 
                        matrix of the dataset is shown.
            Returns:
                    accuracy (float): Accuracy of the model
                    elapsed_time (float): Elapsed time
    """
    model_params = pickle.load(open(file, 'rb'))
    
    k = model_params["kfolds"]
    lr = model_params["learning_rate"]
    
    print('--------------------------------')
    print("MODEL Logistic Regression")

    start = time.time()
    accuracy = Kfold_logistic_regression(dataset, k_folds=k, learning_rate=lr)
    elapsed_time = time.time() - start
                    
    # Print accuracy
    print('Accuracy: %lf %%' % (accuracy))
    print('Time: %lfs' % (elapsed_time))
    print("Parameters:", model_params)
    print('--------------------------------')

    return accuracy, elapsed_time


def score_sklearn_model(filename, dataset):
    """
    Train and score a Scikit-learn SVM model, given a model file.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset
            Returns:
                    accuracy (float): Accuracy of the model
                    elapsed_time (float): Elapsed time
    """
    
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params["kfolds"]

    print('--------------------------------')
    print("MODEL", filename)

    start = time.time()
    accuracy = Kfold_SVM(dataset, k, model_params)
    elapsed_time = time.time() - start

    
    print('Accuracy: %lf %%' % (accuracy*100))
    print('Time: %lfs' % (elapsed_time))
    print("Parameters:", model_params)
    print('--------------------------------')

    return accuracy, elapsed_time


def predictions_SVM(filename, dataset, confusion_matrix = False):
    """
    Trains a SVM model with Kfold and returns the predictions for the whole dataset.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset used for training
                    confusion_matrix (bool): If True, a confusion matrix will be displayed
            Returns:
                    predictions (int list): List with the predictions for each sample     
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    # Load parameters
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params['kfolds']
    
    # Get predictions
    _, predictions = Kfold_SVM(dataset, k, model_params, get_predictions = True)

    if (confusion_matrix):
        visualize_confusion_matrix(predictions, Y)
        
    return predictions


def predictions_logistic(filename, dataset, confusion_matrix = False):
    """
    Trains a Logistic Regression model with Kfold and returns the predictions for the whole dataset.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset used for training
                    confusion_matrix (bool): If True, a confusion matrix will be displayed
            Returns:
                    predictions (int list): List with the predictions for each sample  
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    # Load parameters
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params['kfolds']
    
    # Get predictions
    _, predictions = Kfold_logistic_regression(dataset, get_predictions = True, k_folds=k,
                                                learning_rate=model_params["learning_rate"])
                                                
    if (confusion_matrix):
        visualize_confusion_matrix(predictions, Y)
        
    return predictions
