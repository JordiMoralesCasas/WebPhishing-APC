from generate_features import *
from utils import *

X_train, X_test, y_train, y_test = \
    train_test_split(X_standard, Y, test_size=0.80, random_state=42)

X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
Y_train, Y_test = torch.Tensor(y_train), torch.Tensor(y_test)

X_torch, Y_torch = torch.Tensor(X_standard), torch.Tensor(Y)

data_torch = torch.cat((X_torch, Y_torch), 1)


k_folds = 50
num_epochs = 50
input_dim = X.shape[1] # Independent variables 
output_dim = 1 # Single binary output 
learning_rate = 0.05

results = {}

parameters = []

kfold = KFold(n_splits=k_folds, shuffle=True)
criterion = torch.nn.BCELoss() # Definim el criteri de la funció de cost

for fold, (train_ids, test_ids) in enumerate(kfold.split(reduced_dataset_standard)):
    # Print
    print("\n", f'FOLD {fold}')
    print('--------------------------------')
    
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

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        #print(trainloader)
        for i, data in enumerate(trainloader, 0):
        
            # Get inputs
            #print(data.shape)
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

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving parameters of the linear model
    parameters.append([param.data for name, param in model.named_parameters()]) 

    # Evaluationfor this fold
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

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
    
    # Saving the best model
    if (max(results, key=results.get) == fold):
        save_path = f'../models/logiReg.pth'
    torch.save(model.state_dict(), save_path)
    
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    print("Best fold:", max(results, key=results.get))