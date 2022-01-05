from loading_dataset import *
from hyperparam_search import *


#To start a hyperparameter search for a logistic model, you just need a dataset  (you
# can use any from the ones created in "loading_dataset.py") and a list of
# hyperparameters to search. Give the model a name so you can find it in the "models"
# folder. Below, you can see an example of how to do it:

# Logistic Regression parameters
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
number_of_folds = [2, 5, 10, 20, 30]

hyperparam_search_logistic(reduced_dataset_standard15, learning_rates=learning_rates, 
                            cvfolds=number_of_folds, show_progress=1, 
                            save=True, output_name="TestLogiReg.sav")



# To start a hyperparameter search for an SVM model first indicate what type of search
# you are interested on doing, "grid" (exhaustive grid search) or "random" (Random 
# search, the number of iterations is required as a parameter of the function). Then
# use one of the datasets created in  "loading_dataset.py" and introduce an iterable
# structure for each parameter that you need. Give the model a name so you can find it
# in the "models" folder. The next example shows you how to do it:

n_iter = 200
# SVM parameters
Cs = loguniform(1e-5, 100)
gammas = [0.001*i for i in range(1, 10000)]
coefs = [0.01*i for i in range(1000)]

hyperparam_search_SVM(type_of_search="random", dataset=reduced_dataset_standard15,
                        save=True, output_name="TestSVM.sav",
                        num_iter=10, verbose=10, 
                        kernel="linear", Cs=Cs, cvfolds=3)
