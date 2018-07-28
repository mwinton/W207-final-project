
# coding: utf-8

# # Neural Network Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# In[ ]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data and split class labels into separate array

# In[ ]:


# load dataset from CSV
df = pd.read_csv('data_merged/combined_data_2018-07-18.csv')

# confirm dataset shape looks right
print(df.shape)

# this will show how many non-null values in each column
df.info()

# preview a few rows
df.head()


# In[ ]:


# create y variable with labels
y = df['high_registrations']
y.shape


# ## Train and fit a "naive" model
# For the first model, we'll use all features except SHSAT-related features because they are too correlated with the way we calculated the label.

# In[ ]:


drop_cols = ['dbn',
             'num_shsat_test_takers',
             'offers_per_student',
             'pct_test_takers',
             'high_registrations',
             'school_name',
#              'district',
#              'zip',
            ]

# drop SHSAT-related columns
X = df.drop(drop_cols, axis=1)
print(X.shape)
X.head()


# ### Impute missing values
# 
# The sklearn estimators assume that all values in an array are numerical, and have meaning, so we need to replace `NaN` values.  We choose to use the column means for this imputation.
# 
# > WARNING: this may be problematic for `school_income_estimate` (~2/3 of rows hold nulls)

# In[ ]:


# impute missing values by setting them to the column mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X_imputed = imp.transform(X)

# preview a few rows, post-processed
print(X_imputed.shape)
# confirm col 4 shouldn't be altered; NaNs should be replaced in col 5
print(X_imputed[0:5,4:6])  


# ## One Hot Encoding of categorical explanatory variables
# Columns such as zip code and school district ID, which are integeres should not be fed into an ML model as integers.  Instead, we need to treat them as factors and perform one-hot encoding.

# In[ ]:


# one-hot encode these features as factors
factor_cols = ['district', 'zip']

# get indices for these columns
factor_col_ids = []
for f in factor_cols:
    idx = df.columns.get_loc(f)
    factor_col_ids.append(idx)
factor_col_ids = np.array(factor_col_ids)

print(X_imputed.shape)
ohe_enc = OneHotEncoder(categorical_features=factor_col_ids, handle_unknown='ignore')
X_ohe = ohe_enc.fit_transform(X_imputed)
print(X_ohe.shape)


# ### Split train and test sets
# 
# Split into train (80%) and test (20%) sets

# In[ ]:


# split into training and test sets; make sure to stratify
X_train, X_test, y_train, y_test = util.our_train_test_split(X_ohe, y, stratify=y)

# confirm stratification
print('Frac positive class in training set = %.3f' % (np.sum(y_train==1) / len(y_train)))
print('Frac positive class in test set = %.3f' % (np.sum(y_test==1) / len(y_test)))


# ## Train a "naive" multilayer perceptron model
# This first "naive" model uses all except for the SHSAT-related features, as described above.  We create a pipeline that will be used for k-fold cross-validation.  First, we scale the features, then estimate a multilayer perceptron neural network.

# In[ ]:


# create a pipeline to run these in sequence
n_features = X_train.shape[1]
pipe_clf = make_pipeline(StandardScaler(with_mean=False), 
                   MLPClassifier(hidden_layer_sizes=(n_features,n_features,n_features), max_iter=500))

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds=10
cv_scores = cross_validate(pipe_clf, X_train, y_train, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# ## Train a "race-blind" multilayer perceptron model
# Because we know there's an existing bias problem in the NYC schools, in that the demographics of the test taking population have been getting more homogenous, and the explicit goal of PASSNYC is to make the pool more diverse, we want to train a model that excludes most demographic features.  This would enable us to train a "race-blind" model.  
# 
# ### Preprocess new X_train and X_test datasets

# In[ ]:


race_cols = ['percent_ell',
             'percent_asian',
             'percent_black',
             'percent_hispanic',
             'percent_black__hispanic',
             'percent_white',
             'economic_need_index',
             'school_income_estimate']


# drop additional (demographic) columns
X_race_blind = X.drop(race_cols, axis=1)

# impute missing values by setting them to the column mean
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_race_blind)
X_race_blind_imputed = imp.transform(X_race_blind)

# one-hot encode these features as factors
factor_cols = ['district', 'zip']

# get indices for these columns
factor_col_ids = []
for f in factor_cols:
    idx = X_race_blind.columns.get_loc(f)
    factor_col_ids.append(idx)
factor_col_ids = np.array(factor_col_ids)

# perform one hot encoding
ohe_enc = OneHotEncoder(categorical_features=factor_col_ids, handle_unknown='ignore')
X_race_blind_ohe = ohe_enc.fit_transform(X_race_blind_imputed)

# split into training and test sets; make sure to stratify
X_train, X_test, y_train, y_test = util.our_train_test_split(X_race_blind_ohe, y, stratify=y)

# confirm stratification
print('Frac positive class in training set = %.3f' % (np.sum(y_train==1) / len(y_train)))
print('Frac positive class in test set = %.3f' % (np.sum(y_test==1) / len(y_test)))


# In[ ]:


# create a pipeline to run these in sequence
n_features = X_train.shape[1]
pipe_clf = make_pipeline(StandardScaler(with_mean=False), 
                   MLPClassifier(hidden_layer_sizes=(n_features,n_features,n_features), max_iter=500))

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds=10
cv_scores = cross_validate(pipe_clf, X_train, y_train, cv=k_folds, scoring=['accuracy','f1'])


# In[ ]:


util.print_cv_results(cv_scores)


# ## Final test set accuracy

# In[ ]:


# y_predict = mlp.predict(X_test)

# print(confusion_matrix(y_test,y_predict))
# print(classification_report(y_test,y_predict))

