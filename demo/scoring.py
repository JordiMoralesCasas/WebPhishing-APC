from loading_dataset import *
from score_model import *

# Test a logistic regression model
score_logistic_torch("../models/TestLogiReg.sav", reduced_dataset_standard15)

# Test SVM with linear kernel
score_sklearn_model("../models/TestSVM.sav", reduced_dataset_standard15)
