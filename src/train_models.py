import sys
sys.path.insert(1, 'helpers')

from generate_features import *
from utils import *


def Kfold_logistic_regression(dataset, k_folds=2, num_epochs=5, learning_rate=0.05, get_predictions = False):
    """
    Train a PyTorch Logistic Regression model by doing a KFold, given a model file.
            Parameters:
                    dataset (pandas dataframe): Dataset
                    k_folds (int): Number of folds.
                    num_epochs (int): Number of epochs during the training
                    learning_rate (float): Learning rate
                    get_predictions (bool): If True, the predictions for all the dataset will be
                       calculated for each fold and the combined predicted is returned
            Returns:
                    predictions (int list): List with the predictions for each sample. Only if get_predictions is True
                    avg_accuracy (float): Average accuracy of the trained model
                    avg_recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    avg_f1 (float): F1 score or F-measure. Harmonic mean of the precision 
    """
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    input_dim = X.shape[1] # Independent variables 
    output_dim = 1 # Single binary output

    # The criterion for the cost function uses the Binary Cross Entropy between the target and
    # the input probabilities. Very common in binary classification tasks.
    criterion = torch.nn.BCELoss()

    if (get_predictions):
            # Create an array to store all the predictions. Also initialize a torch DataLoader.
            preds = np.zeros([dataset.shape[0], k_folds])
            data_loader = torch.utils.data.DataLoader(dataset.to_numpy())

    # Starting the KFold
    kfold = KFold(n_splits=k_folds, shuffle=True)

    accuracies = []
    recalls = []
    f1_scores = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset.to_numpy(), 
                        batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset.to_numpy(),
                        batch_size=10, sampler=test_subsampler)
        
        # Create the model
        model = LogisticRegression(input_dim,output_dim)
        
        # Initialize optimizer (Stochastic Gradient Decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs = data[:,:-1].float()
                targets = data[:, -1].reshape((inputs.shape[0],1)).float()

                # Zero the gradients and perform forward pass
                optimizer.zero_grad()
                outputs = model(inputs)

                # Compute loss and perform backward pass
                loss = criterion(outputs, targets)
                loss.backward()

                # Perform optimization
                optimizer.step()
                
        # Evaluation for this fold
        correct, total, true_positive, false_negative = 0, 0, 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs = data[:,:-1].float()
                targets = data[:, -1].reshape((inputs.shape[0],1)).float()

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                total += targets.size(0)
                correct += np.sum(outputs.round().detach().numpy() == targets.detach().numpy())

                # Get number of true positive and false negative
                for i in range(len(targets.detach().numpy())):
                    if (targets.detach().numpy()[i] == 1 and outputs.round().detach().numpy()[i] == 1):
                        true_positive += 1
                    if (targets.detach().numpy()[i] == 1 and outputs.round().detach().numpy()[i] == 0):
                        false_negative += 1
                
            accuracies.append((correct / total))
            recalls.append(true_positive/(true_positive + false_negative))

        if (get_predictions):
            # Get the prediction for each sample
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(data_loader, 0):
                    # Get inputs
                    inputs = data[:,:-1].float()
                    targets = data[:, -1].reshape((inputs.shape[0],1)).float()
                    
                    # Generate outputs
                    outputs = model(inputs)

                    # Save prediction
                    preds[i, fold] = outputs.round().detach().numpy()[0][0]

    # Compute all the scores
    avg_accuracy = sum(accuracies)/len(accuracies)
    avg_recall = sum(recalls)/len(recalls)
    avg_f1 = 2*(avg_accuracy * avg_recall) / (avg_accuracy + avg_recall)

    if (get_predictions):
        # Combined prediction of all the folds
        predictions = [np.argmax(np.bincount(preds[i, :].astype('int'))) for i in range(X.shape[0])]
        return predictions, avg_accuracy, avg_recall, avg_f1
    else:
        return avg_accuracy, avg_recall, avg_f1



def Kfold_SVM(dataset, k_folds, model_params, get_predictions = False):
    """
    Train a scikit-learn SVM model by doing a KFold, given a model file.
            Parameters:
                    dataset (pandas dataframe): Dataset
                    k_folds (int): Number of folds.
                    model_params (dictionary): Set of hyperparameters for the SVM
                    get_predictions (bool): If True, the predictions for all the dataset will be
                        calculated for each fold and the combined predicted is returned
            Returns:
                    predictions (int list): List with the predictions for each sample. Only if get_predictions is True
                    avg_accuracy (float): Average accuracy of the trained model
                    avg_recall (float): Recall score (Ability of the classifier to find all the positive samples)
                    avg_f1 (float): F1 score or F-measure. Harmonic mean of the precision 
                    
    """

    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    # Create the model
    model = svm.SVC(kernel = model_params['kernel'],
                    C=model_params['C'],
                    gamma = model_params['gamma'],
                    coef0 = model_params['coef0'], 
                    degree = model_params['degree'],
                    max_iter = -1)

    if (get_predictions):
        # Create an array to store all the predictions. Also initialize a fold counter.
        preds = np.zeros([k_folds, dataset.shape[0]])
        fold = 0

    # Starting the KFold
    kf = KFold(n_splits=k_folds)

    acc_kfold = []
    recall_kfold = []
    f1_kfold = []
    for train_index , test_index in kf.split(X):
        X_train , X_test = X[train_index,:],X[test_index,:]
        Y_train , Y_test = Y[train_index] , Y[test_index]
        
        model.fit(X_train,Y_train)
        pred_values = model.predict(X_test)
        acc_kfold.append(accuracy_score(pred_values , Y_test))
        recall_kfold.append(recall_score(pred_values, Y_test))
        f1_kfold.append(f1_score(pred_values, Y_test))

        if (get_predictions):
            # Save all the predictions for this fold
            preds[fold] = model.predict(X)
            fold += 1

    # Compute all the scores
    avg_accuracy = sum(acc_kfold)/k_folds
    avg_recall = sum(recall_kfold)/k_folds
    avg_f1 = sum(f1_kfold)/k_folds

    if (get_predictions):
        # Combined prediction of all the folds
        predictions = [np.argmax(np.bincount(preds[:, i].astype('int'))) for i in range(X.shape[0])]
        return predictions, avg_accuracy, avg_recall, avg_f1
    else:
        return avg_accuracy, avg_recall, avg_f1
