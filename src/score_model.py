from imports import *
from utils import *
from generate_features import *


"""def score_torch_model(filename, dataset, show_confusion_matrix = False):
    input_dim = dataset.shape[1] # Independent variables 
    output_dim = 1 # Single binary output 

    model = pickle.load(open(filename, 'rb'))

    testloader = torch.utils.data.DataLoader(
                          dataset.to_numpy(),
                          batch_size=dataset.shape[0])

    correct, total = 0.0, 0.0
    with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs = data[:,:-1].float()
                targets = data[:, -1].reshape((inputs.shape[0],1)).float()

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)

                correct += np.sum(outputs.round().detach().numpy() == targets.detach().numpy())
            
            if (show_confusion_matrix):
                visualize_confusion_matrix(outputs.round().detach().numpy(), targets.detach().numpy())
            
            # Print accuracy
            print('--------------------------------')
            print("MODEL", filename)
            print('Accuracy: %lf %%' % (100.0 * correct / total))
            print('--------------------------------')"""


def score_logistic_torch(dataset, num_folds, show_confusion_matrix = False):
    input_dim = dataset.shape[1] # Independent variables 
    output_dim = 1 # Single binary output 

    preds = np.zeros([dataset.shape[0], num_folds])

    testloader = torch.utils.data.DataLoader(
                            dataset.to_numpy())

    for fold in range(num_folds):
        filename = '../models/LogiRegFolds/modelLogiReg'+ str(fold) +'.sav'
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
                
    if (show_confusion_matrix):
                    visualize_confusion_matrix(predictions, dataset.values[:,-1])

    # Print accuracy
    print('--------------------------------')
    print('Accuracy: %lf %%' % ((dataset.values[:,-1] == predictions).sum()/len(predictions)*100))
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
    score_logistic_torch(reduced_dataset_standard, 5)

    score_sklearn_model("../models/modelLinear.sav", reduced_dataset_standard)

    score_sklearn_model("../models/modelRbf.sav", reduced_dataset_standard)

    score_sklearn_model("../models/modelSigmoid.sav", reduced_dataset_standard)

    """score_sklearn_model("../models/modelPoly1.sav", reduced_dataset_standard)

    score_sklearn_model("../models/modelPoly2.sav", reduced_dataset_standard)

    score_sklearn_model("../models/modelPoly3.sav", reduced_dataset_standard)
    
    score_sklearn_model("../models/modelPoly4.sav", reduced_dataset_standard)"""



    if (False):
        score_sklearn_model("../models/archive/modelLinear.sav", reduced_dataset_standard)

        score_sklearn_model("../models/archive/modelRbf.sav", reduced_dataset_standard)

        score_sklearn_model("../models/archive/modelSigmoid.sav", reduced_dataset_standard)

        score_sklearn_model("../models/archive/modelPoly1.sav", reduced_dataset_standard)

        score_sklearn_model("../models/archive/modelPoly2.sav", reduced_dataset_standard)

        score_sklearn_model("../models/archive/modelPoly3.sav", reduced_dataset_standard)