import sys
sys.path.insert(1, 'helpers')

from generate_features import *
from imports import *
from pca import *
from train_models import *


def hyperparam_search_logistic(dataset, learning_rates=[0.05], cvfolds=[10], num_epochs=10, save = False,  output_name = "out.sav", show_progress=10):
    """
    Exhaustive search over specified hyperparameter values for a PyTorch Logistic Regression model. The best model will be saved.
            Parameters:
                    dataset (pandas dataframe): Dataset
                    learning_rate (float list): List of Learning Rates
                    cvfolds (int list): List Number of folds
                    num_epochs (int): Number of epochs during the training
                    save (bool): If True, the best set of parameters will be saved into a file
                    output_name (string): Name of the file where the best model will be saved
                    show_progress (int): How often show a message with the progress (number of iterations)
            Returns:
                    best_params (dictionary): Best set of parameters found during the search
    """    
    print("MODEL Logistic regression\nStarting search:")

    best_params = {'learning_rate': 0, 'kfolds' : 0}
    
    #Starting the search
    max_accuracy = 0
    idx = 0
    for lr in learning_rates:
        for k in cvfolds:
            if idx % show_progress == 0:
                print("Progress: "+str(idx)+"/"+str(len(learning_rates)*len(cvfolds)))
            
            # KFold for the current set of hyperparameters
            current_accuracy = Kfold_logistic_regression(dataset, k_folds=k, num_epochs=num_epochs, learning_rate=lr)

            if (current_accuracy > max_accuracy):
                max_accuracy = current_accuracy
                best_params["learning_rate"] = lr
                best_params["kfolds"] = k
            idx += 1
    
    print("\nFinal scoring (\"accuracy\"):", max_accuracy, "\n")

    if (save):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/' + output_name, 'wb'))

    return best_params

def hyperparam_search_SVM(type_of_search, dataset, kernel, num_iter=100, Cs=[1], gammas=[1], coefs=[1], deg=1, cvfolds=2,
            save = False, output_name="out.sav", verbose=0):
    """
    Exhaustive search over specified hyperparameter values for a scikit-learn SVM model. The best model will be saved.
            Parameters:
                    type_of_search: If "grid", perform an exhaustive grid search. If "random", perform a randomized search.
                    dataset (pandas dataframe): Dataset
                    kernel (string): Kernel to be used in the SVM algorithm ("linear", "poly", "rbf", "sigmoid")
                    num_iter (int): Number of iterations for the randomized search
                    Cs (float list): List of C values (regularization parameter)
                    gammas (float list): List of gamma values. Kernel coefficient for "rbf", "poly" or sigmoid
                    coefs (float list): List of independent terms for "poly" and "sigmoid" kernel functions
                    deg (int): Degree of the polynomial kernel function ("poly").
                    cvfolds (int list): List Number of folds
                    save (bool): If True, the best set of hyperparameters will be saved into a file
                    output_name (string): Name of the file where the best parameters will be saved
                    verbose (float): How often show a message with the progress
            Returns:
                    best_params (dictionary): Best set of hyperparameters found during the search
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    print("MODEL SVM\nKernel:", kernel)

    # Dictionary with all the parameters
    parameters = [{'kernel' : [kernel],
                   'C' : Cs,
                   'gamma' : gammas,
                   'coef0' : coefs,
                   'degree' : [deg]}]
    # Create an empty model
    estimator = svm.SVC()

    # Starting the search
    if (type_of_search == "grid"):
        search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring="accuracy", cv=cvfolds, n_jobs=-1, verbose=verbose)
    elif (type_of_search == "random"):
        search = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, scoring="accuracy", n_iter=num_iter, cv=cvfolds, n_jobs=-1, verbose=verbose)

    # Final results
    search.fit(X, Y)
    print("\nFinal scoring (\"accuracy\"):", search.score(X, Y), "\n")

    best_params = search.best_params_

    # Add the number of folds as an aditonal hyperparameter
    best_params["kfolds"] = cvfolds

    if (save):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/' + output_name, 'wb'))

    return best_params