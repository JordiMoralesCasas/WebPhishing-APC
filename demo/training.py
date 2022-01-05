from loading_dataset import *
from hyperparam_search import *

# Logistic Regression parameters
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
number_of_folds = [2, 5, 10, 20, 30]

hyperparam_search_logistic(reduced_dataset_standard15, learning_rates=learning_rates, 
                            cvfolds=number_of_folds, show_progress=1, 
                            save=True, output_name="TestLogiReg.sav")


n_iter = 200
# SVM parameters
Cs = loguniform(1e-5, 100)
gammas = [0.001*i for i in range(1, 10000)]
coefs = [0.01*i for i in range(1000)]

hyperparam_search_SVM(type_of_search="random", dataset=reduced_dataset_standard15, kernel="linear",
                        save=True, output_name="TestSVM.sav",
                        num_iter=10, verbose=10, 
                        Cs=Cs, cvfolds=3)
