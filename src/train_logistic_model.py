from generate_features import *
from utils import *


def train_logistic_regression_kfold(dataset, k_folds=2, num_epochs=1, learning_rate=0.05, save=False):
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y, test_size=0.80, random_state=42)

    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    Y_train, Y_test = torch.Tensor(y_train), torch.Tensor(y_test)

    X_torch, Y_torch = torch.Tensor(X), torch.Tensor(Y)

    input_dim = X.shape[1] # Independent variables 
    output_dim = 1 # Single binary output

    accuracies = []
    kfold = KFold(n_splits=k_folds, shuffle=True)
    criterion = torch.nn.BCELoss() # Definim el criteri de la funció de cost

    for fold, (train_ids, test_ids) in enumerate(kfold.split(reduced_dataset_standard)):
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        reduced_dataset_standard.to_numpy(), 
                        batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        reduced_dataset_standard.to_numpy(),
                        batch_size=10, sampler=test_subsampler)
        
        # Init the neural network
        model = LogisticRegression(input_dim,output_dim)
        
        # Initialize optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Definim el mètode per reduir el cost
        
        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs = data[:,:-1].float()
                targets = data[:, -1].reshape((inputs.shape[0],1)).float()

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()
                
        # Evaluation for this fold
        correct, total = 0, 0
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

            accuracies.append(100.0 * (correct / total))
        
        # Saving the model
        if (save):
            save_path = '../models/LogiRegFolds/modelLogiReg'+ str(fold) +'.sav'
            pickle.dump(model, open(save_path, 'wb'))
        
    average_accuracy = sum(accuracies)/len(accuracies)
    return average_accuracy

def hyperparam_search_logistic(dataset, num_epochs=10, learning_rates=[0.05], number_of_folds=[10], show_progress=10):
    print("MODEL Logistic regression\nStarting search:")
    max_accuracy = 0
    best_params = {'learning_rate': 0, 'num_folds' : 0}
    idx = 0
    for lr in learning_rates:
        for k in number_of_folds:
            if idx % show_progress == 0:
                    print("Progress: "+str(idx)+"/"+str(len(learning_rates)*len(number_of_folds)))
            current_accuracy = train_logistic_regression_kfold(dataset, k_folds=k, num_epochs=num_epochs, learning_rate=lr)

            if (current_accuracy > max_accuracy):
                max_accuracy = current_accuracy
                best_params["learning_rate"] = lr
                best_params["num_folds"] = k
            idx += 1
    
    print("Search finished. Saving model:")
    train_logistic_regression_kfold(dataset, k_folds=best_params["num_folds"], num_epochs=num_epochs, learning_rate=best_params["learning_rate"], save=True)
    print("DONE")

if __name__ == "__main__":

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
    number_of_folds = [2, 5, 10, 20, 30]
    #hyperparam_search_logistic(reduced_dataset_standard, learning_rates=learning_rates, number_of_folds=number_of_folds, show_progress=1)