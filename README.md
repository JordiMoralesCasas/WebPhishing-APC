# Pràctica Kaggle APC UAB 2021-22
#### Author: Jordi Morales Casas
#### DATASET (kaggle): [Web page Phishing Detection Dataset](https://www.kaggle.com/shashwatwork/web-page-phishing-detection-dataset)
## **Description of the data**
This dataset, as described on the Kaggle website, is formed mostly by data that has been extracted directly from the structure and syntax of different URLs, as well as from content of their corresponding websites and information collected by querying external services.

The dataset consists of 11429 unique samples with 89 attributes:
  - **31** represent categorical data. In fact, all of them are binary variables. The objective variable (*status*) is included here.
  - **1** that contains the whole URL.
  - **55** are numerical data, most in the form of counts or averages.

### Objective
The main goal is to train a supervised learning model that can detect if a website is being used for phishing or not.


## Experiments
[comment]: <> (Durant aquesta pràctica hem realitzat diferents experiments.)
So far I have used two different algorithms for the analysis:

- A logistic regression model implemented with PyTorch

- Different SVM models with all the different predefined kernels, implemented with scikit-learn:

    - Linear kernel.
    - RBF (Radial Basis Function) kernel.
    - Sigmoid kernel.
    - Polynomial kernel (up to 4th degree).




### Preprocessing
[comment]: <> (Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?)
The initial dataset didn't required a lot of preprocessingn regarding missing values or useless data. The only removed feature was the *URL* since all the information that can be extracted from it is already collect in the rest of columns.

Also, luckily since we want to perform a binary classification, the samples were evenly split between those labeled as *phishing* and *legitimate* (5715 of each, exactly 50%), so in that sense there wasn't any problem related to unbalanced data.

Once that was covered, the first consideration was if feature scaling would represent an improvement in the classification task or not (although I knew this was one of the most important preprocessing step, I thought that it would be interesting to see the difference). After training some models for both standardized and non-standardized data, the results showed that the non-standardized models not only got a significantly lower accuracy than the standardized ones (at best, the first ones got 64.2% accuracy while the worst standardized got 86.51%), but also the time needed for the training in some cases was so large (it didn't finish after several hours) that I couldn't find a model for the SVM algorithm with linear and polynomial (1st, 2nd and 3rd degree) kernels.

The last experiment regarding preprocessing was seeing if reducing the dimensions of the data would be beneficial in any way. By this I mean that a model, even with a slightly worse accuracy, may be still worth it because the training time is more convenient.

For this purpose I have studied two methods for reducing the dimensions. The first one is performing a feature selection based on the Mutual Information between each independent variable and the objective variable. This is a great score function for data that mixes categorical and non-categorical variables, as well as being model neutral (can be applied to different types of ML models) and relatively fast.

Two new datasets were created with feature selection, one with 15 attributes and the other with 30. I chose arbitrarily these number because I was interested in studying the effect of a significant reduction of features (Remember that we began with 87 features). After some tests, the results showed that for both cases, the accuracies were pretty close to the model without feature selection (usually a few tenths behind) and the training time was much better,
specially with the case with 30 features, consistently improving by 40% to 60% the original time.

The other method of reducing the dimensionality of the dataset is by performing a Principal Component Analysis (PCA). By taking into account the variance of each principal component we can find the number of dimensions (components) that we want to keep in order to maintaince a certain amount of the original variance. I decided to keep the 90% of that variance, and thus I had to keep 50 components.

The results regarding precision were very close to the original dataset, but the training time varies greatly depending on the model, in some cases lasting several minutes, and if we also add the time spent doing the PCA we are left with hardly any room for improvement in any case.

### Model summary

The accuracy scores and training times for all the previous experiments are gathered in the *scores.txt* file. Also the same models are saved in the *models* folder and can be loaded using the functions provided (See [Demo](#demo1)).

| Model | Details | Hiperparameters | Accuracy  | Time |
| -- | -- | -- | -- | -- |
| Logistic Regression | 30 features (FS) | learning_rate: 0.1,	kfolds: 2 | 93.43 % | 2.5324s |
| SVM | Linear kernel, full dataset | C: 0.4993382731012802, kfolds: 3 | 94.716 % | 4.1421s |
| SVM | RBF kernel, 15 features (FS) | C: 5.8413889806557835,  gamma: 0.084, kfolds: 3 | 94.436 % | 3.9144s |
| SVM | Polynomial kernel (2nd deg.), full dataset  | C: 0.000277029357932836, coef0: 4.63, gamma: 5.409, kfolds: 10 | 95.617 % | 22.2996s |
| SVM | Polynomial kernel (2nd deg.), PCA  | C: 0.0001725847091854454,  coef0: 5.44, gamma: 7.47, kfolds: 10 | 95.311 % | 16.2763s |
| SVM | Polynomial kernel (3nd deg.), 30 features (FS) | C: 0.008761422316284174,  coef0: 6.74, gamma: 0.434, kfolds: 10 | 95.188 % | 14.8542s |

## <a name="demo1"></a>Demo 
There are already trained models saved in this repository. They can be loaded by executing the following command, alongside with their score:

``` python demo/scoring.py ```

For training and saving new models, you can run this script:

``` python demo/train_models.py ```

## Conclusions
[comment]: <> (El millor model que s'ha aconseguit ha estat... En comparació amb l'estat de l'art i els altres treballs que hem analitzat....)
[comment]: <> (## Idees per treballar en un futur)
[comment]: <> (Crec que seria interesant indagar més en...)
[comment]: <> (## Llicencia)
[comment]: <> (El projecte s’ha desenvolupat sota llicència ZZZz.)