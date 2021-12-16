from math import e
from imports import *
from generate_features import *


def hyperparam_search_SVM(type_of_search, kernel, dataset, output_name="out.sav",
                            num_iter=100, Cs=[1], gammas=[1], coefs=[1], deg=1, cvfolds=2, verbose=0):
    #Loading dataset
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    print("MODEL SVM\nKernel:", kernel)
    print("Starting search (" + type_of_search + "):\n")

    parameters = [{'kernel' : [kernel],
                   'C' : Cs,
                   'gamma' : gammas,
                   'coef0' : coefs,
                   'degree' : [deg]}]
    estimator = svm.SVC()

    if (type_of_search == "grid"):
        search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring="accuracy", cv=cvfolds, n_jobs=-1, verbose=verbose)
    elif (type_of_search == "random"):
        search = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, scoring="accuracy", n_iter=num_iter, cv=cvfolds, n_jobs=-1, verbose=verbose)
    search.fit(X, Y)

    print("\nFinal scoring (\"accuracy\"):", search.score(X, Y), "\n")

    #Training model with the hyperparameters resulting from the search
    b_params = search.best_params_
    model = svm.SVC(kernel = b_params['kernel'],
                    C=b_params['C'],
                    gamma = b_params['gamma'],
                    coef0 = b_params['coef0'], 
                    degree = b_params['degree'],
                    max_iter = -1)

    #Saving model to pickle file
    model.fit(X, Y)
    pickle.dump(model, open('../models/' + output_name, 'wb'))



if __name__ == "__main__":
    #TODO: precomputed kernel

    Cs = [0.01*i for i in range(1, 1000)]
    #hyperparam_search_SVM("grid", "linear", reduced_dataset_standard, Cs=Cs)

    Cs = loguniform(1e-5, 100)
    gammas = [0.001*i for i in range(1, 10000)]
    coefs = [0.01*i for i in range(1000)]

    """hyperparam_search_SVM("random", "rbf", reduced_dataset_standard,
                            output_name="modelRbf.sav",
                            num_iter=2000, verbose=10, 
                            Cs=Cs, gammas=gammas, cvfolds=3)

    hyperparam_search_SVM("random", "sigmoid", reduced_dataset_standard,
                            output_name="modelSigmoid.sav",
                            num_iter=2000, verbose=10,
                            Cs=Cs, gammas=gammas, cvfolds=3)"""

    hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                            output_name="modelPoly1.sav",
                            num_iter=2000, verbose=10,
                            Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

    """hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                            output_name="modelPoly2.sav",
                            num_iter=2000, verbose=10,
                            Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=3)

    hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                            output_name="modelPoly3.sav",
                            num_iter=2000, verbose=10,
                            Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=3)
    
    hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                            output_name="modelPoly4.sav",
                            num_iter=2000, verbose=10,
                            Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=3)"""

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=1, show_progress = 10)

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=2, show_progress = 10)

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=3, show_progress = 10)"""