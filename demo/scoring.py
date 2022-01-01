from loading_dataset import *
from score_model import *


score_logistic_torch("../models/TestLogiReg.sav", reduced_dataset_standard15)

score_sklearn_model("../models/TestSVM.sav", reduced_dataset_standard15)
