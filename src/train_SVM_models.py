from imports import *
from generate_features import *


def hyperparam_search_SVM(model_name, dataset, Cs=[1], gammas=[1], coefs=[1], deg=1, show_progress = 100, max_iter = -1):
    X = dataset.values[:,:-1]
    Y = dataset.values[:,-1]

    X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.1, random_state=1)

    accuracies = np.zeros([len(Cs)*len(gammas)*len(coefs), 4])
    max_accur = 0
    
    if (model_name == "poly"):
        filename = '../models/model' + model_name.capitalize() + str(deg) + '.sav'
    else:
        filename = '../models/model' + model_name.capitalize() + '.sav'
    
    print("MODEL SVM\nKernel:", model_name, "\nStarting search")
    idx = 0
    for coef in coefs:
        for g in gammas:
            for c in Cs:
                if idx % show_progress == 0:
                    print("Progress: "+str(idx)+"/"+str(len(Cs)*len(gammas)*len(coefs)))

                model = svm.SVC(kernel = model_name, C=c, gamma = g, coef0 = coef, degree = deg, max_iter = max_iter)
                model.fit(X_train,y_train.flatten())
                predict = model.predict(X_test)

                accuracies[idx][0] = (predict == y_test.flatten()).sum()/predict.size #Accuracy
                accuracies[idx][1] = c
                accuracies[idx][2] = g
                accuracies[idx][3] = coef

                current_accuracy = accuracies[np.argmax(accuracies[:, 0])][0]
                if (max_accur < current_accuracy):
                    max_accur = current_accuracy
                    pickle.dump(model, open(filename, 'wb'))
                
                idx += 1
    print("DONE\n")

if __name__ == "__main__":
    Cs = [0.025*i for i in range(1, 41)] + [0.5*i for i in range(2, 21)]
    gammas = [1/reduced_dataset_standard.shape[1]] + [0.025*i for i in range(1, 41)]
    coefs = [0.25*i for i in range(4)] + [i for i in range(6)]

    
    #Cs = [0.01*i for i in range(1, 100)] + [0.2*i for i in range(5, 51)]
    #hyperparam_search_SVM("linear", reduced_dataset_standard, Cs=Cs, show_progress = 20)

    hyperparam_search_SVM("rbf", reduced_dataset_standard, Cs=Cs, gammas=gammas, show_progress = 50)

    #hyperparam_search_SVM("sigmoid", reduced_dataset_standard, Cs=Cs, gammas=gammas, show_progress = 10)

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=1, show_progress = 10)

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=2, show_progress = 10)

    #hyperparam_search_SVM("poly", reduced_dataset_standard, Cs=Cs, gammas=gammas, coefs=coefs, deg=3, show_progress = 10)