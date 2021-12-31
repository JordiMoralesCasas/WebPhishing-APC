from loading_dataset import *
from score_model import *


#score_sklearn_model("../models/RbfFullStd.sav", Full_dataset_standard)
predictions_SVM("../models/RbfFullStd.sav", Full_dataset_standard, confusion_matrix=True)

#score_logistic_torch("../models/LogiRegFullStd.sav", Full_dataset_standard)
#predictions_logistic("../models/LogiRegPCA.sav", pca_dataset, confusion_matrix = True)


"""print(" -  Logistic Regression Models")
score_logistic_torch("../models/LogiRegReduced15.sav", reduced_dataset_standard15)
score_logistic_torch("../models/LogiRegReduced30.sav", reduced_dataset_standard30)
score_logistic_torch("../models/LogiRegFullStd.sav", Full_dataset_standard)
score_logistic_torch("../models/LogiRegFull.sav", Full_dataset)
score_logistic_torch("../models/LogiRegPCA.sav", pca_dataset)

print(" -  SVM Linear Models")
score_sklearn_model("../models/LinearReduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/LinearReduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/LinearFullStd.sav", Full_dataset_standard)
#score_sklearn_model("../models/LinearFull.sav", Full_dataset)
score_sklearn_model("../models/LinearPCA.sav", pca_dataset)

print(" -  SVM RBF Models")
score_sklearn_model("../models/RbfReduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/RbfReduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/RbfFullStd.sav", Full_dataset_standard)
score_sklearn_model("../models/RbfFull.sav", Full_dataset)
score_sklearn_model("../models/RbfPCA.sav", pca_dataset)

print(" -  SVM Sigmoid Models")
score_sklearn_model("../models/SigmoidReduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/SigmoidReduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/SigmoidFullStd.sav", Full_dataset_standard)
score_sklearn_model("../models/SigmoidFull.sav", Full_dataset)
score_sklearn_model("../models/SigmoidPCA.sav", pca_dataset)

print(" -  SVM Polynomial 1 Models")
score_sklearn_model("../models/Poly1Reduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/Poly1Reduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/Poly1FullStd.sav", Full_dataset_standard)
#score_sklearn_model("../models/Poly1Full.sav", Full_dataset)
score_sklearn_model("../models/Poly1PCA.sav", pca_dataset)

print(" -  SVM Polynomial 2 Models")
score_sklearn_model("../models/Poly2Reduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/Poly2Reduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/Poly2FullStd.sav", Full_dataset_standard)
#score_sklearn_model("../models/Poly2Full.sav", Full_dataset)
score_sklearn_model("../models/Poly2PCA.sav", pca_dataset)

print(" -  SVM Polynomial 3 Models")
score_sklearn_model("../models/Poly3Reduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/Poly3Reduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/Poly3FullStd.sav", Full_dataset_standard)
#score_sklearn_model("../models/Poly3Full.sav", Full_dataset)
score_sklearn_model("../models/Poly3PCA.sav", pca_dataset)

print(" -  SVM Polynomial 4 Models")
score_sklearn_model("../models/Poly4Reduced15.sav", reduced_dataset_standard15)
score_sklearn_model("../models/Poly4Reduced30.sav", reduced_dataset_standard30)
score_sklearn_model("../models/Poly4FullStd.sav", Full_dataset_standard)
score_sklearn_model("../models/Poly4Full.sav", Full_dataset)
score_sklearn_model("../models/Poly4PCA.sav", pca_dataset)"""
