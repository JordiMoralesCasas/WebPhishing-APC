from utils import *
from generate_features import *

input_dim = X_standard.shape[1] # Independent variables 
output_dim = 1 # Single binary output 

model =  LogisticRegression(input_dim,output_dim)
model.load_state_dict(torch.load(f'../models/logiReg.pth'))
model.eval()

testloader = torch.utils.data.DataLoader(
                      reduced_dataset_standard.to_numpy(),
                      batch_size=10)

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
            #correct += (predicted == targets).sum().item()
            #print(outputs, targets.shape)
            
            correct += np.sum(outputs.round().detach().numpy() == targets.detach().numpy())

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (8, 100.0 * correct / total))
        print('--------------------------------')