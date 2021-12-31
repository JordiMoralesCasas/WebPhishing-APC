from loading_dataset import *
from hyperparam_search import *

# SVM parameters
n_iter = 100
Cs = loguniform(1e-5, 100)
gammas = [0.001*i for i in range(1, 10000)]
coefs = [0.01*i for i in range(1000)]

# Logistic Regression parameters
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
number_of_folds = [2, 5, 10, 20, 30]


######## Training without feature selection (15 features)

hyperparam_search_logistic(reduced_dataset_standard15, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="LogiRegReduced15.sav")

hyperparam_search_SVM("random", "linear", reduced_dataset_standard15,
                        output_name="LinearReduced15.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)
hyperparam_search_SVM("random", "rbf", reduced_dataset_standard15,
                        output_name="RbfReduced15.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", reduced_dataset_standard15,
                        output_name="SigmoidReduced15.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard15,
                        output_name="Poly1Reduced15.sav",
                        num_iter=50, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=4)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard15,
                        output_name="Poly2Reduced15.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=10)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard15,
                        output_name="Poly3Reduced15.sav",
                        num_iter=1, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=10)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard15,
                        output_name="Poly4Reduced15.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=10)

######## Training without feature selection (30 features)

hyperparam_search_logistic(reduced_dataset_standard30, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="LogiRegReduced30.sav")

hyperparam_search_SVM("random", "linear", reduced_dataset_standard30,
                        output_name="LinearReduced30.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)

hyperparam_search_SVM("random", "rbf", reduced_dataset_standard30,
                        output_name="RbfReduced30.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", reduced_dataset_standard30,
                        output_name="SigmoidReduced30.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard30,
                        output_name="Poly1Reduced30.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard30,
                        output_name="Poly2Reduced30.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=10)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard30,
                        output_name="Poly3Reduced30.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=10)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard30,
                        output_name="Poly4Reduced30.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=10)

######## Training without feature selection (Standardized)

hyperparam_search_logistic(Full_dataset_standard, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="LogiRegFullStd.sav")

hyperparam_search_SVM("random", "linear", Full_dataset_standard,
                        output_name="LinearFullStd.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)

hyperparam_search_SVM("random", "rbf", Full_dataset_standard,
                        output_name="RbfFullStd.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", Full_dataset_standard,
                        output_name="SigmoidFullStd.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset_standard,
                        output_name="Poly1FullStd.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset_standard,
                        output_name="Poly2FullStd.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=10)

hyperparam_search_SVM("random", "poly", Full_dataset_standard,
                        output_name="Poly3FullStd.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=10)

hyperparam_search_SVM("random", "poly", Full_dataset_standard,
                        output_name="Poly4FullStd.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=10)


######## Training without feature selection (Non-standardized)



hyperparam_search_logistic(Full_dataset, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="LogiRegFull.sav")

hyperparam_search_SVM("random", "linear", Full_dataset,
                        output_name="LinearFull.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)

hyperparam_search_SVM("random", "rbf", Full_dataset,
                        output_name="RbfFull.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", Full_dataset,
                        output_name="SigmoidFull.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly1Full.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly2Full.sav",
                        num_iter=1, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=10)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly3Full.sav",
                        num_iter=1, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=5)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly4Full.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=10)

#Training after performing a pca with a number of components that ensures a retained variance greater than 0.90

hyperparam_search_logistic(pca_dataset, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="LogiRegPCA.sav")

hyperparam_search_SVM("random", "linear", pca_dataset,
                        output_name="LinearPCA.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)

hyperparam_search_SVM("random", "rbf", pca_dataset,
                        output_name="RbfPCA.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", pca_dataset,
                        output_name="SigmoidPCA.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly1PCA.sav",
                        num_iter=30, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly2PCA.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=10)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly3PCA.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=10)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly4PCA.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=10)