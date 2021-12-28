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

Luckily for the purpose of the analysis, the dataset is evenly split between samples labeled as *phishing* and *legitimate* (5715 of each, exactly 50%).

## Experiments
[comment]: <> (Durant aquesta pràctica hem realitzat diferents experiments.)

### Preprocessing
[comment]: <> (Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?)

### Model summary
| Model | Hiperparameters | Score | Time |
| -- | -- | -- | -- |
| XX | -- | -- | -- |
| XX | -- | -- | -- |
| XX | -- | -- | -- |

## Demo
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