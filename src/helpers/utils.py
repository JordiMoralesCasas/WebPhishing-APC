from imports import *

# Logistic regression model (PyTorch)
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

def logistic_func(x):
    """
    Logistic function
            Parameters:
                    x (flaot): input
    """
    return 1 / (1 + np.exp(-x))


def visualize_confusion_matrix(y_pred, y_real):
    """
    Show the confusion matrix
            Parameters:
                    y_pred (numpy array): Array of predictions
                    y_real (numpy array): Array of real values
    """
    cm = confusion_matrix(y_real, y_pred)
    plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot = True, fmt = 'g', xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def make_meshgrid(x, y, h=.02):
    """
    Create a mesh of points to plot in
            Parameters:
                    x (numpy array: data to base x-axis meshgrid on
                    y (numpy array): data to base y-axis meshgrid on
                    h (float): stepsize for meshgrid, optional
            Returns
                    xx (numpy array): n-dimensional array
                    yy (numpy array): n-dimensional array
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """
    Plot the decision boundaries for a classifier.
            Parameters:
                    ax : matplotlib axes object
                    clf (SVM model): a classifier
                    xx (numpy array): meshgrid ndarray
                    yy (numpy array): meshgrid ndarray
                    params (dictionary): dictionary of params to pass to contourf, optional
            return:
                    out (graphic object): output graphic
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def SVM_decision_boundary(dataset, filename, column_pairs):
    """
    Visual 2D representation the decision boundary of a SVM model.
            Parameters:
                    dataset (pandas dataframe): Dataset used for training
                    filename (string): file that contains a saved model
                    column_pairs (list of string lists): A list containing lists with
                        the pairs of features that want to be visualized
    """
    # Load params
    model_params = pickle.load(open(filename, 'rb'))
    
    kernel = model_params['kernel']
    C = model_params['C']
    gamma = model_params['gamma']
    coef0 = model_params['coef0']
    degree = model_params['degree']

    # create SVM model
    model = svm.SVC(kernel=kernel, C=C, gamma=gamma, coef0 = coef0, degree=degree)
    
    # Create subplots
    plt.close('all')
    fig, sub = plt.subplots(1, len(column_pairs), figsize=(20,5))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    if (len(column_pairs) == 1):
        aux = [sub]
    else:
        aux = sub.flatten()
    
    # Iterate over each pair of features
    for pair, ax in zip(column_pairs, aux):
        X = dataset[pair].values
        Y = dataset.values[:,-1]
        
        model = model.fit(X, Y)
        
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        
        plot_contours(ax, model, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(pair[0])
        ax.set_ylabel(pair[1])
        ax.set_xticks(())
        ax.set_yticks(())

    plt.show()