
# coding: utf-8

# # K Nearest Neighbors
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton
# 
# #### W207-4-Summer 2018 Final Project
# 
# [Return to project overview](final_project_overview.ipynb)

# In this notebook, we attempt to classify the PASSNYC data via K-Nearest Neighbors algorithm.  The idea is to use "K-nearest neighbors" classifier to "learn" schools that have high number of SHSAT registrations, and use that to predict on a test set.
# 
# ***
# 
# ### Reading data
# Let us do some initial imports and set up the data.

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
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


# #### Feature selection
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


# ***
# 
# ## K-Nearest Neighbors Classification
# 
# #### Default run
# We will now run KNN prediction on the dataset, with the default K value (=5).

# In[4]:


scaler = StandardScaler().fit(perf_train_data_nonull)
rescaledX = scaler.transform(perf_train_data_nonull)
y = train_labels
clf = KNeighborsClassifier()

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# #### Searching for best $k$
# We get accuracy of 82% and F1 score of 0.58.  Let us experiment with various values of $k$ to see which gives the best results.

# In[5]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=k_folds, n_jobs=-1, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

best_k = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of neighbors: %d" % best_k)


# The best value for number of neighbors is $k=3$.  Let us get the scores for this value of $k$.

# In[6]:


# Do k-fold cross-validation, collecting both "test" accuracy and F1 
clf = KNeighborsClassifier(n_neighbors=best_k)
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# We get accuracy of 85% and F1 of 0.64.  Let us run the above KNN model on the test set.

# In[7]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier(n_neighbors=best_k))
pipeline.fit(perf_train_data_nonull, train_labels)
predicted_labels = pipeline.predict(perf_test_data_nonull)
knn_score_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
knn_score_f1 = metrics.f1_score(test_labels, predicted_labels)

print("On the test set, the model has an accuracy of {:.2f}% and an F1 score of {:.2f}."
     .format(knn_score_accuracy*100, knn_score_f1))


# We can see accuracy of 87% on test set, and F1 of 0.73.
# 
# We will later report the above two values: one for cross-validation, and another for test set.

# *** 
# 
# ### KNN with select features
# 
# We will now attempt to do some feature selection, followed by running KNN.

# In[8]:


pipeline = make_pipeline(StandardScaler(), 
                         SelectFromModel(ExtraTreesClassifier(random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, y)
selected_features = pipeline.steps[1][1].get_support()
selected_cols = perf_train_data_nonull.columns[selected_features].values.tolist()
print("Selected feature columns: %s" % selected_cols)


# #### Searching for best $k$
# 
# Let us also find the best number of neighbors for this subset of features.

# In[9]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=k_folds, n_jobs=-1, scoring='f1')

perf_train_data_nonull_sel_cols = selected_cols
perf_train_data_nonull_sel = perf_train_data_nonull[perf_train_data_nonull_sel_cols]
estimator.fit(perf_train_data_nonull_sel, y)

best_k = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of neighbors: %d" % best_k)


# We will use this to run cross-validation on the model.

# In[10]:


scaler = StandardScaler().fit(perf_train_data_nonull_sel)
rescaledX_sel = scaler.transform(perf_train_data_nonull_sel)
clf = KNeighborsClassifier(n_neighbors=best_k)

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
cv_scores = cross_validate(clf, rescaledX_sel, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# F1 score falls a little to 0.59.  Let us look at how it performs on test set.

# In[11]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier(n_neighbors=best_k))
pipeline.fit(perf_train_data_nonull_sel, y)
perf_test_data_nonull_sel = perf_test_data_nonull[perf_train_data_nonull_sel_cols]
predicted_labels = pipeline.predict(perf_test_data_nonull_sel)
knn_score_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
knn_score_f1 = metrics.f1_score(test_labels, predicted_labels)

print("On the test set, the model has an accuracy of {:.2f}% and an F1 score of {:.2f}."
     .format(knn_score_accuracy*100, knn_score_f1))


# We see that the model does not do well on test set either.  Nevertheless, we will report this as a second run.

# ***
# 
# ### KNN with reduced dimensions
# 
# We will next attempt to reduce dimensions via PCA, followed by KNN.
# 
# First, we will attempt to find the best number of components.

# In[12]:


# generate plot of variance explained vs # principale components
util.get_num_pcas(perf_train_data_nonull, var_explained=0.9)


# We can see that the first 3 components already explain more than 70% of variance.  The slope of the graph goes down after this, indicating that remaining components are not as informative.
# 
# #### Searching for best $k$
# 
# Let us run GridSearch on both PCA components and K, to see if we can get a better model.

# In[13]:


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
print("Best no. of PCA components: %d, neighbors: %d" % 
      (best_pca_components,
       best_k))


# We find that PCA with 8 components, followed by KNN with 7 neighbors is the best combination.

# In[14]:


# Do k-fold cross-validation, collecting both "test" accuracy and F1 
pipeline = make_pipeline(StandardScaler(), 
                         PCA(random_state=207, n_components=best_pca_components),
                         KNeighborsClassifier(n_neighbors=best_k))

cv_scores = cross_validate(pipeline, perf_train_data_nonull, train_labels, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# With this combination, we get accuracy of 84% and F1 score of 0.63.  Let's run the above model on test set and determine our model scores.

# In[15]:


pipeline = make_pipeline(StandardScaler(), 
                         PCA(random_state=207, n_components=best_pca_components),
                         KNeighborsClassifier(n_neighbors=best_k))
pipeline.fit(perf_train_data_nonull, train_labels)
predicted_labels = pipeline.predict(perf_test_data_nonull)
lr_score_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
lr_score_f1 = metrics.f1_score(test_labels, predicted_labels)

print("On the test set, the model has an accuracy of {:.2f}% and an F1 score of {:.2f}."
     .format(lr_score_accuracy*100, lr_score_f1))


# We get accuracy of 88% and a rather good F1 score of 0.73.  The accuracy improves a tiny bit on test set, but falls a bit in 5-fold cross-validation.
# 
# Let us summarize our three runs of KNN so far.
# 
# ***
# 
# ### Summary
# 
# Model | CV Accuracy | (95% CI) | CV F1 | (95% CI) | Test Set Accuracy | Test Set F1
# :---|:---:|:---:|:---:|:---:|:---:|:---:
# K-Nearest Neighbors (Full Model) | 0.849 | (0.721, 0.976) | 0.637 | (0.328, 0.946) | 0.87 | 0.73
# K-Nearest Neighbors (Top n Features) | 0.841 | (0.756, 0.926) | 0.586 | (0.376, 0.795) | 0.84 | 0.65
# K-Nearest Neighbors (PCA, most features) | 0.841 | (0.714, 0.967) | 0.630 | (0.357, 0.903) | 0.88 | 0.73
# 

# ***
# 
# ### List of schools for PASSNYC
# 
# Let us look at what schools the model classified as positive, but were actually negative.  These are the schools we should target, because the model thinks they should have high SHSAT registrations, but in reality they do not.
# 
# We will use the last model run above (PCA, most features) to generate the school list.

# In[16]:


pipeline = make_pipeline(StandardScaler(),
                         PCA(n_components=best_pca_components, random_state=207),
                         KNeighborsClassifier(n_neighbors=best_k))

fp_df = util.run_model_get_ordered_predictions(pipeline, train_data, test_data,
                                               perf_train_data_nonull, perf_test_data_nonull,
                                               train_labels, test_labels)


# Now that we have the false positives, we will obtain a ranking of the schools that we can provide to PASSNYC.

# In[17]:


df_passnyc = util.create_passnyc_list(fp_df, train_data, test_data,
                                 train_labels, test_labels)
# Write to CSV
df_passnyc.to_csv('results/results.knn.csv')

df_passnyc

