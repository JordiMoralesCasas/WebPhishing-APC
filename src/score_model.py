from imports import *
from utils import *
from generate_features import *


def score_torch_model(filename, dataset, show_confusion_matrix = False):
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
    print('--------------------------------')


if __name__ == "__main__":
    score_torch_model('../models/modelLogiReg.pth', reduced_dataset_standard)

    #score_sklearn_model("../models/modelLinear.sav", reduced_dataset_standard)

    #score_sklearn_model("../models/modelRbf.sav", reduced_dataset_standard)

    #score_sklearn_model("../models/modelSigmoid.sav", reduced_dataset_standard)

    #score_sklearn_model("../models/modelPoly1.sav", reduced_dataset_standard)

    #score_sklearn_model("../models/modelPoly2.sav", reduced_dataset_standard)

    #score_sklearn_model("../models/modelPoly3.sav", reduced_dataset_standard)