from loading_dataset import *
from score_model import *

print(' -  "15 best features" models')

#score_logistic_torch(reduced_dataset_standard, 30, output_name="reduced", folder_path="../models/LogiRegReduced")

score_sklearn_model("../models/LinearReduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/RbfReduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/SigmoidReduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/Poly1Reduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/Poly2Reduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/Poly3Reduced.sav", reduced_dataset_standard)

score_sklearn_model("../models/Poly4Reduced.sav", reduced_dataset_standard)


print(' -  Full dataset models')

"""score_logistic_torch(Full_dataset, 30, output_name="full", folder_path="../models/LogiRegFull")

score_sklearn_model("../models/LinearFull.sav", Full_dataset)

score_sklearn_model("../models/RbfFull.sav", Full_dataset)

score_sklearn_model("../models/SigmoidFull.sav", Full_dataset)

score_sklearn_model("../models/Poly1Full.sav", Full_dataset)

score_sklearn_model("../models/Poly2Full.sav", Full_dataset)

score_sklearn_model("../models/Poly3Full.sav", Full_dataset)

score_sklearn_model("../models/Poly4Full.sav", Full_dataset)"""



print(' -  PCA dataset models')

"""score_logistic_torch(pca_dataset, 30, output_name="pca", folder_path="../models/LogiRegPCA")

score_sklearn_model("../models/LinearPCA.sav", pca_dataset)

score_sklearn_model("../models/RbfPCA.sav", pca_dataset)

score_sklearn_model("../models/SigmoidPCA.sav", pca_dataset)

score_sklearn_model("../models/Poly1PCA.sav", pca_dataset)

score_sklearn_model("../models/Poly2PCA.sav", pca_dataset)

score_sklearn_model("../models/Poly3PCA.sav", pca_dataset)

score_sklearn_model("../models/Poly4PCA.sav", pca_dataset)"""