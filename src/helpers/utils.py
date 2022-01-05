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
