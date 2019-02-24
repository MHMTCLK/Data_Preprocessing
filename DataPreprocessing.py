# DATA PREPROCESSING


## IMPORT LIBRARY ##

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## IMPORT DATA ##

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


## MISSING DATA (Imputer)##

# Taking care of missing data
# first option is removed this line
# second option is replace mean, strategy=mean
# third option is replace median, strategy=median
# fourth option is replace most frequent (like single mode), strategy=most_frequent
# axis=0 for the column based, axis=1 for the row based
from sklearn.preprocessing import Imputer #help(Imputer)
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


## CATEGORICAL DATA (LabelEncoder,OneHotEncoder)##

# categorical data with LabelEncoder
# label encoder is used your data include just 2 different variables
# or you want to sort according to important level
# e.g larger greater than medium and also medium greater than small
# but you have to be carefull because for the modelling, 
# sometimes small bigger than others
# sometimes medium bigger than others for the important level.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
labelEncoder = labelEncoder.fit(X[:, 0])
X[:,0] = labelEncoder.transform(X[:, 0])
#X[:,0] = labelEncoder.fit_transform(X[:,0])

y = labelEncoder.fit_transform(y)

# categorical data with OneHotEncoder
# it is different from the LabelEncoder
# is transformed all categories as 0 and 1 by the it
# so if you have two more than categories like above
# firstly you have to apply LabelEncoder
# after that you have to apply OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
oneHotEncoder = oneHotEncoder.fit(X)
X = oneHotEncoder.transform(X).toarray()
#X = labelEncoder.fit_transform(X).toarray()

#y have two categories so apply OneHotEncoder is unnecessary 


## SPLIT DATA (train,test) ##

# splitting data for learning and test
# as a result create a model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #is used 42 in general


## FEATURE SCALE ##

# must do features scaling
# because all variables must be same interval
# if it is not, your model is not good and won't getting better
# for example age between 20 and 50
# salary between 30k and 70k
# age is not important for your math model (eucliden: distance between two point)
# so you must do scale for same interval and important level
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #help(StandardScaler)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #is fitted in above so just we should do transform

# y is unnecessary for scaling, because it is a categorical variable
# and also this is classification problem.
# most popular scaling types:  MinMaxScaler,  StandardScaler,  Normalizer,  Binarizer
# please look at this website for the reference : https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/