from loading_dataset import *
from train_SVM_models import *
from train_logistic_model import *

n_iter = 100

Cs = loguniform(1e-5, 100)
gammas = [0.001*i for i in range(1, 10000)]
coefs = [0.01*i for i in range(1000)]

#### SVM

######## Training without feature selection

"""hyperparam_search_SVM("random", "linear", reduced_dataset_standard,
                        output_name="LinearReduced.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, cvfolds=3)

hyperparam_search_SVM("random", "rbf", reduced_dataset_standard,
                        output_name="RbfReduced.sav",
                        num_iter=n_iter, verbose=10, 
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "sigmoid", reduced_dataset_standard,
                        output_name="SigmoidReduced.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                        output_name="Poly1Reduced.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                        output_name="Poly2Reduced.sav",
                        num_iter=5, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=3)"""

"""hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                        output_name="Poly3Reduced.sav",
                        num_iter=1, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=3)

hyperparam_search_SVM("random", "poly", reduced_dataset_standard,
                        output_name="Poly4Reduced.sav",
                        num_iter=1, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=3)"""

######## Training without feature selection

"""hyperparam_search_SVM("random", "linear", Full_dataset,
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
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly3Full.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=3)

hyperparam_search_SVM("random", "poly", Full_dataset,
                        output_name="Poly4Full.sav",
                        num_iter=2, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=3)"""


#Training after performing a pca with a number of components that ensures a retained variance greater than 0.90

"""hyperparam_search_SVM("random", "linear", pca_dataset,
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
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=1, cvfolds=3)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly2PCA.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=2, cvfolds=3)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly3PCA.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=3, cvfolds=3)

hyperparam_search_SVM("random", "poly", pca_dataset,
                        output_name="Poly4PCA.sav",
                        num_iter=n_iter, verbose=10,
                        Cs=Cs, gammas=gammas, coefs=coefs, deg=4, cvfolds=3)"""



#### LOGISTIC FOREST


learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
number_of_folds = [2, 5, 10, 20, 30]

#learning_rates = [0.01]
#number_of_folds = [2]


hyperparam_search_logistic(reduced_dataset_standard, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="reduced", folder_path="../models/LogiRegReduced")

"""hyperparam_search_logistic(Full_dataset, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="full", folder_path="../models/LogiRegFull")

hyperparam_search_logistic(pca_dataset, learning_rates=learning_rates, 
                            number_of_folds=number_of_folds, show_progress=1, 
                            output_name="pca", folder_path="../models/LogiRegPCA")"""