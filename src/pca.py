from imports import *
from generate_features import *



def perform_pca(dataset, auto = False , n_comp=X.shape[1], obj_variance = 0.95):
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    X_torch = torch.tensor(X)
    Y_torch = torch.tensor(Y)

    if (auto):
        S = torch.pca_lowrank(X_torch, q=X_torch.shape[1], center=True, niter=100)[1]
        n_comp = [(k+1, j) for k, j in enumerate(np.array([sum(S.numpy()[:i]/sum(S.numpy())) for i in range(1,X_torch.shape[1]+1)]) > obj_variance) if (j == True)][0][0]


    V = torch.pca_lowrank(X_torch, q=n_comp, center=False, niter=100)[2]   
    
    pca = torch.matmul(X_torch, V[:, :n_comp]).numpy()
    pca_dataset = pd.DataFrame(pca, columns=[ "pc"+str(i) for i in range(1, n_comp+1)])
    
    pca_dataset = pca_dataset.assign(status=Y)
    
    return pca_dataset


if __name__ == "__main__":
    #pca2 = perform_pca(Full_dataset, Y, n_comp = 2)

    #pca3 = perform_pca(Full_dataset, Y, n_comp = 3)

    pca = perform_pca(Full_dataset, auto=True, obj_variance=0.90)