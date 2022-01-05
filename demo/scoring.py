from loading_dataset import *
from score_model import *

# To get the scores of a saved model introduce its path and the dataset that 
# used for training it. Below you can find two examples

# Test Logistic Regression model
score_logistic_torch("../models/TestLogiReg.sav", reduced_dataset_standard15)

# Test SVM with linear kernel
score_SVM_sklearn("../models/TestSVM.sav", reduced_dataset_standard15)