
# coding: utf-8

# # K Nearest Neighbors
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton
# 
# #### W207-4-Summer 2018 Final Project
# 
# [Return to project overview](final_project_overview.ipynb)

# In this notebook, we attempt to classify the PASSNYC data via K-Nearest Neighbors algorithm.
# 
# ### Reading data
# Let us do some initial imports and set up the data.

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from sklearn.decomposition import PCA

import util

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

k_folds = 5
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Get train-test split
train_data, test_data, train_labels, test_labels = util.read_data()

print("Train data shape: %s" % str(train_data.shape))
print("Test data shape: %s" % str(test_data.shape))
train_data.head()


# ### Feature selection
# 
# We will now select some features from the above dataset.
# 
# We will ignore some categorical variables and variables that are highly correlated with outcome variable.

# In[3]:


drop_cols = [
    # non-numeric
    'dbn',
    # correlated with outcome variable
    'num_shsat_test_takers',
    'offers_per_student',
    'pct_test_takers',
    # demographic or correlated with demographics
    'school_name',
    'zip',
    'district',
    # too many nulls
    'school_income_estimate',
]
perf_train_data = train_data.drop(drop_cols, axis=1)
perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())

perf_test_data = test_data.drop(drop_cols, axis=1)
perf_test_data_nonull = perf_test_data.fillna(perf_test_data.mean())

# Tried this out; got lower F1
# train_data_ohe, test_data_ohe = util.get_dummies(perf_train_data_nonull,
#                                                  perf_test_data_nonull,
#                                                  factor_cols=['district'])
# train_data_ohe.head()

#perf_test_data_nonull = perf_test_data.drop('district', axis=1)


# ### K-Nearest Neighbors Classification
# 
# We will now run KNN prediction on the dataset, with the default K value (=5).

# In[4]:


scaler = MinMaxScaler().fit(perf_train_data_nonull)
rescaledX = scaler.transform(perf_train_data_nonull)
y = train_labels
clf = KNeighborsClassifier()

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# We get accuracy of 82% and F1 score of 0.58.  Let us experiment with various values of $k$ to see which gives the best results.

# In[5]:


pipeline = make_pipeline(MinMaxScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=k_folds, n_jobs=-1, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

print("Best no. of neighbors: %d (with best f1: %.3f)" % 
      (estimator.best_params_['kneighborsclassifier__n_neighbors'], 
       estimator.best_score_))
best_k = estimator.best_params_['kneighborsclassifier__n_neighbors']


# The best F1 score is 0.59 at $k=5$.

# ### KNN with select features
# 
# We will now attempt to do some feature selection, followed by running KNN.

# In[6]:


pipeline = make_pipeline(MinMaxScaler(), 
                         SelectFromModel(ExtraTreesClassifier(random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, y)
selected_features = pipeline.steps[1][1].get_support()
selected_cols = perf_train_data_nonull.columns[selected_features].values.tolist()
print("Selected feature columns: %s" % selected_cols)


# In[7]:


perf_train_data_nonull_sel_cols = selected_cols
perf_train_data_nonull_sel = perf_train_data_nonull[perf_train_data_nonull_sel_cols]
scaler = MinMaxScaler().fit(perf_train_data_nonull_sel)
rescaledX = scaler.transform(perf_train_data_nonull_sel)
y = train_labels
clf = KNeighborsClassifier(n_neighbors=best_k)

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# F1 score falls a tiny bit to 0.58.  We can ignore this set and use the original set instead.

# ### KNN with reduced dimensions
# 
# We will next attempt to reduce dimensions via PCA, followed by KNN.
# 
# First, we will attempt to find the best number of components.

# In[8]:


# generate plot of variance explained vs # principale components
util.get_num_pcas(perf_train_data_nonull, var_explained=0.9)


# We can see that the first 3 components already explain more than 70% of variance.  The slope of the graph goes down after this, indicating that remaining components are not as informative.
# 
# Let us run GridSearch on both PCA components and K, to see if we can get a better model.

# In[9]:


pipeline = make_pipeline(StandardScaler(), 
                         PCA(random_state=207),
                         KNeighborsClassifier())

n_components = list(range(1, 12))
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(pca__n_components=n_components,
                             kneighborsclassifier__n_neighbors=n_neighbors),
                             cv=k_folds, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

best_pca_components = estimator.best_params_['pca__n_components']
best_k = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of PCA components: %d, neighbors: %d (with best f1: %.3f)" % 
      (best_pca_components,
       best_k,
       estimator.best_score_))


# PCA with 8 components, followed by KNN with 7 neighbors, gives us F1-score that's up by 0.05: earlier, it was 0.58, now it's 0.63.

# ### False Positives
# 
# Let us look at what schools the model classified as positive, but were actually negative.  These are the schools we should target, because the model thinks they should have high SHSAT registrations, but in reality they do not.

# In[10]:


pipeline = make_pipeline(StandardScaler(),
                         PCA(n_components=best_pca_components, random_state=207),
                         KNeighborsClassifier(n_neighbors=best_k))

fp_df = util.run_model_get_ordered_predictions(pipeline, train_data, test_data,
                                      perf_train_data_nonull, perf_test_data_nonull,
                                      train_labels, test_labels)


# Now that we have the false positives, we will obtain a ranking of the schools that we can provide to PASSNYC.

# In[11]:


df_passnyc = util.create_passnyc_list(fp_df, train_data, test_data,
                                 train_labels, test_labels)
# Write to CSV
#df_passnyc.to_csv('results/results.knn.csv')

df_passnyc

