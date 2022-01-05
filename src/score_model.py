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
                    recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    f1_score (float): F1 score or F-measure. Harmonic mean of the precision 
                    elapsed_time (float): Elapsed time
                    model_params (dictionary): Set of hyperparameters
    """
    # Load parameters
    model_params = pickle.load(open(file, 'rb'))
    k = model_params["kfolds"]
    lr = model_params["learning_rate"]
    
    # Training
    start = time.time()
    accuracy, recall, f1_score = Kfold_logistic_regression(dataset, k_folds=k, learning_rate=lr)
    elapsed_time = time.time() - start

    return accuracy, recall, f1_score, elapsed_time, model_params


def score_SVM_sklearn(filename, dataset):
    """
    Train and score a Scikit-learn SVM model, given a model file.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset
            Returns:
                    accuracy (float): Accuracy of the model
                    recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    f1_score (float): F1 score or F-measure. Harmonic mean of the precision 
                    elapsed_time (float): Elapsed time
                    model_params (dictionary): Set of hyperparameters 
    """
    # Load parameters
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params["kfolds"]

    start = time.time()
    accuracy, recall, f1_score = Kfold_SVM(dataset, k, model_params)
    elapsed_time = time.time() - start

    return accuracy, recall, f1_score, elapsed_time, model_params


def predictions_SVM(filename, dataset, confusion_matrix = False):
    """
    Trains a SVM model with Kfold and returns the predictions for the whole dataset.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset used for training
                    confusion_matrix (bool): If True, a confusion matrix will be displayed
            Returns:
                    predictions (int list): List with the predictions for each sample
                    accuracy (float): Accuracy of the model
                    recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    f1score (float): F1 score or F-measure. Harmonic mean of the precision  
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    # Load parameters
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params['kfolds']
    
    # Get predictions
    predictions, _, _, _ = Kfold_SVM(dataset, k, model_params, get_predictions = True)

    if (confusion_matrix):
        visualize_confusion_matrix(predictions, Y)

    # Compute the accuracy score 
    accuracy = accuracy_score(Y, predictions)

    # Compute the recall score    
    recall = recall_score(Y, predictions)

    # Compute the f1 score
    f1score = f1_score(Y, predictions)

    return predictions, accuracy, recall, f1score


def predictions_logistic(filename, dataset, confusion_matrix = False):
    """
    Trains a Logistic Regression model with Kfold and returns the predictions for the whole dataset.
            Parameters:
                    file (string): file that contains a saved model
                    dataset (pandas dataframe): Dataset used for training
                    confusion_matrix (bool): If True, a confusion matrix will be displayed
            Returns:
                    predictions (int list): List with the predictions for each sample
                    accuracy (float): Accuracy of the model
                    recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    f1score (float): F1 score or F-measure. Harmonic mean of the precision  
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    # Load parameters
    model_params = pickle.load(open(filename, 'rb'))
    k = model_params['kfolds']
    
    # Get predictions
    predictions, _, _, _  = Kfold_logistic_regression(dataset, get_predictions = True, k_folds=k,
                                                learning_rate=model_params["learning_rate"])
                                                
    if (confusion_matrix):
        visualize_confusion_matrix(predictions, Y)
    
    # Compute the accuracy score 
    accuracy = accuracy_score(Y, predictions)

    # Compute the recall score    
    recall = recall_score(Y, predictions)
    
    # Compute the f1 score
    f1score = f1_score(Y, predictions)

    return predictions, accuracy, recall, f1score
