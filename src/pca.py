import sys
sys.path.insert(1, 'helpers')

#from imports import *
from generate_features import *



def perform_pca(dataset, auto = False , n_comp=None, obj_variance = 1):
    """
    Performs a Principal Component Analysis over the data. The number of components that
    the dataset is reduced to can be set manually, with a given value, or automatically,
    by looking to mantain an objective variance.
            Parameters:
                    dataset (pandas dataframe): Dataset
                    auto (bool): Boolean variable. If True, the number of components 
                        is given by the user. If False, the best number of components is
                        found automatically.
                    n_comp (int): Number of components to keep. If "auto" is equal to
                        True, this value is ignored.
                    obj_variance (float): Objective variance that the processed dataset
                        have to maintain. If "auto" is equal to False, this value is ignored
            Returns:
                    pca_dataset (pandas dataframe): Dataset after applying PCA
    """
    # Convert data to torch tensors
    X, Y = dataset.values[:,:-1], dataset.values[:,-1]
    X_torch, Y_torch = torch.tensor(X), torch.tensor(Y)

    # After performing a PCA, an approximation of a singular value decomposition is returned
    USV = torch.pca_lowrank(X_torch, q=X_torch.shape[1], center=True, niter=100)

    if (auto):
        S = USV[1] # Provides information about the variance of the different variables

        # The number of principal components to keep is the lowest number that ensures a variance >= obj_variance.
        n_comp = [(k+1, j) for k, j in enumerate(np.array([sum(S.numpy()[:i]/sum(S.numpy())) for i in range(1,X_torch.shape[1]+1)]) > obj_variance) if (j == True)][0][0]
    elif(n_comp == None):
        # If any n_comp is specified, all the components will be kept
        n_comp = X.shape[1]


    # The following operation projects the data to the first n_comp principal components
    pca = torch.matmul(X_torch, USV[2][:, :n_comp]).numpy()

    # Merge all the data into a single dataframe 
    pca_dataset = pd.DataFrame(pca, columns=[ "pc"+str(i) for i in range(1, n_comp+1)])
    pca_dataset = pca_dataset.assign(status=Y)
    
    return pca_dataset