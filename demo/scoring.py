from loading_dataset import *
from score_model import *

# To get the scores of a saved model introduce its path and the dataset that 
# used for training it. It is also possible to view the confusion matrix for
# the dataset. If you are testing a SVM model, you can visualize its the
# decision boundary. Below you can find two examples:


# Test Logistic Regression model

accuracy, recall, f1_score, elapsed_time, model_params = score_logistic_torch("../models/TestLogiReg.sav", reduced_dataset_standard15)
print(" - Logistic regression model")
print("Accuracy score:", accuracy)
print("Recall score:", recall)
print("F1 score:", f1_score)
print("Training time:", elapsed_time)
print("Model parameters: ", model_params)

# Show confusion matrix for all the samples
predictions_logistic("../models/TestLogiReg.sav", reduced_dataset_standard15, confusion_matrix=True)


# Test SVM with linear kernel

accuracy, recall, f1_score, elapsed_time, model_params = score_SVM_sklearn("../models/TestSVM.sav", reduced_dataset_standard15)
print("\n\n - SVM model")
print("Accuracy score:", accuracy)
print("Recall score:", recall)
print("F1 score:", f1_score)
print("Training time:", elapsed_time)
print("Model parameters: ", model_params)

# Show confusion matrix for all the samples
predictions_SVM("../models/TestSVM.sav", reduced_dataset_standard15, confusion_matrix=True)
# Show decision boundary for a given pair of features
SVM_decision_boundary(reduced_dataset_standard15, "../models/TestSVM.sav", [['google_index','page_rank'], ["nb_www","domain_age"]])
