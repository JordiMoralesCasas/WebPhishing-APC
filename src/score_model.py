from imports import *
from utils import *
from generate_features import *

from pca import *

def score_logistic_torch(dataset, num_folds, show_confusion_matrix = False, output_name = "out", folder_path = "../model"):
    input_dim = dataset.shape[1] # Independent variables 
    output_dim = 1 # Single binary output 

    preds = np.zeros([dataset.shape[0], num_folds])

    testloader = torch.utils.data.DataLoader(
                            dataset.to_numpy())
    start = time.time()
    for fold in range(num_folds):
        filename = folder_path + "/" + output_name + str(fold) +'.sav'
        model = pickle.load(open(filename, 'rb'))

        with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    
                    # Get inputs
                    inputs = data[:,:-1].float()
                    targets = data[:, -1].reshape((inputs.shape[0],1)).float()
                    #print(inputs, targets)

                    # Generate outputs
                    outputs = model(inputs)

                    # Set total and correct
                    preds[i, fold] = outputs.round().detach().numpy()[0][0]

    predictions = [np.bincount(preds[i].astype('int')).argmax() for i in range(dataset.shape[0])]
    accuracy = (dataset.values[:,-1] == predictions).sum()/len(predictions)*100

    end = time.time()
    if (show_confusion_matrix):
                    visualize_confusion_matrix(predictions, dataset.values[:,-1])

    # Print accuracy
    print('--------------------------------')
    print("MODEL Logistic Regression")
    print('Accuracy: %lf %%' % (accuracy))
    print('Time: %lfs' % (end - start))
    print('--------------------------------')


def score_sklearn_model(filename, dataset, show_confusion_matrix = False):
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]
    model = pickle.load(open(filename, 'rb'))
    predict = model.predict(X)
    
    if (show_confusion_matrix):
        visualize_confusion_matrix(predict, Y.flatten())
    print('--------------------------------')
    print("MODEL", filename)
    print('Accuracy: %lf %%' % ((predict == Y.flatten()).sum()/predict.size*100))
    #print(model.get_params())
    print('--------------------------------')


if __name__ == "__main__":

    print(' -  "15 best features" models')

    score_logistic_torch(reduced_dataset_standard, 2, output_name="reduced", folder_path="../models/LogiRegReduced")

    score_sklearn_model("../models/LinearReduced.sav", reduced_dataset_standard)

    score_sklearn_model("../models/RbfReduced.sav", reduced_dataset_standard)

    score_sklearn_model("../models/SigmoidReduced.sav", reduced_dataset_standard)

    score_sklearn_model("../models/Poly1Reduced.sav", reduced_dataset_standard)

    score_sklearn_model("../models/Poly2Reduced.sav", reduced_dataset_standard)

    score_sklearn_model("../models/Poly3Reduced.sav", reduced_dataset_standard)
    
    score_sklearn_model("../models/Poly4Reduced.sav", reduced_dataset_standard)


    print(' -  Full dataset models')

    score_logistic_torch(Full_dataset, 2, output_name="full", folder_path="../models/LogiRegFull")

    score_sklearn_model("../models/LinearFull.sav", Full_dataset)

    score_sklearn_model("../models/RbfFull.sav", Full_dataset)

    score_sklearn_model("../models/SigmoidFull.sav", Full_dataset)

    score_sklearn_model("../models/Poly1Full.sav", Full_dataset)

    score_sklearn_model("../models/Poly2Full.sav", Full_dataset)

    score_sklearn_model("../models/Poly3Full.sav", Full_dataset)
    
    score_sklearn_model("../models/Poly4Full.sav", Full_dataset)


    print(' -  PCA dataset models')

    pca_dataset = perform_pca(Full_dataset, auto=True, obj_variance=0.90) 

    score_logistic_torch(pca_dataset, 2, output_name="pca", folder_path="../models/LogiRegPCA")

    score_sklearn_model("../models/LinearPCA.sav", pca_dataset)

    score_sklearn_model("../models/RbfPCA.sav", pca_dataset)

    score_sklearn_model("../models/SigmoidPCA.sav", pca_dataset)

    score_sklearn_model("../models/Poly1PCA.sav", pca_dataset)

    score_sklearn_model("../models/Poly2PCA.sav", pca_dataset)

    score_sklearn_model("../models/Poly3PCA.sav", pca_dataset)
    
    score_sklearn_model("../models/Poly4PCA.sav", pca_dataset)

    