import os

import numpy as np
import pandas as pd
import math
import seaborn as sns
import plotly.express as px
import scipy.stats
from matplotlib import pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


# Torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

#Sklearn
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# Set working directory to current location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)