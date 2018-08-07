
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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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
y = train_labels
perf_train_data = train_data.drop(drop_cols, axis=1)
perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())
perf_test_data = test_data.drop(drop_cols, axis=1)
perf_test_data_nonull = perf_test_data.fillna(perf_test_data.mean())


# ***
# 
# ## K-Nearest Neighbors Classification
# 
# We will now run KNN on the dataset.  We will run it in three ways:
# 
# 1. Run with all features except for the dropped ones above
# 2. Run with a few features, based on a feature selection algorithm
# 3. Run PCA, then run with all features except for the dropped ones above
# 
# ### 1/3. KNN with most features
# We will now run KNN prediction on the dataset.  We will use grid search to get the best value for the hyper-parameter $k$.

# #### Searching for best $k$
# Let us experiment with various values of $k$ to see which gives the best results.

# In[ ]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=k_folds, n_jobs=-1, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

best_k_all_features = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of neighbors: %d" % best_k_all_features)


# The best value for number of neighbors is $k=3$.  Let us get the scores for this value of $k$.

# In[ ]:


scaler = StandardScaler().fit(perf_train_data_nonull)
rescaledX = scaler.transform(perf_train_data_nonull)
# Do k-fold cross-validation, collecting both "test" accuracy and F1 
clf = KNeighborsClassifier(n_neighbors=best_k_all_features)
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# We get accuracy of 85% and F1 of 0.64.

# *** 
# 
# ### 2/3. KNN with select features
# 
# We will now attempt to do some feature selection, followed by running KNN.

# In[ ]:


pipeline = make_pipeline(StandardScaler(), 
                         SelectFromModel(ExtraTreesClassifier(random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, y)
selected_features = pipeline.steps[1][1].get_support()
selected_cols = perf_train_data_nonull.columns[selected_features].values.tolist()
print("Selected feature columns: %s" % selected_cols)


# #### Searching for best $k$
# 
# Let us also find the best number of neighbors for this subset of features.

# In[ ]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=k_folds, n_jobs=-1, scoring='f1')

perf_train_data_nonull_sel_cols = selected_cols
perf_train_data_nonull_sel = perf_train_data_nonull[perf_train_data_nonull_sel_cols]
estimator.fit(perf_train_data_nonull_sel, y)

best_k_some_features = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of neighbors: %d" % best_k_some_features)


# We will use this to run cross-validation on the model.

# In[ ]:


scaler = StandardScaler().fit(perf_train_data_nonull_sel)
rescaledX_sel = scaler.transform(perf_train_data_nonull_sel)
clf = KNeighborsClassifier(n_neighbors=best_k_some_features)

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
cv_scores = cross_validate(clf, rescaledX_sel, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# F1 score falls a little to 0.59.

# ***
# 
# ### 3/3. KNN with reduced dimensions
# 
# We will next attempt to reduce dimensions via PCA, followed by KNN.
# 
# First, we will attempt to find the best number of components.

# In[ ]:


# generate plot of variance explained vs # principale components
util.get_num_pcas(perf_train_data_nonull, var_explained=0.9)


# We can see that the first 3 components already explain about 40% of variance.  The slope of the graph goes down after this.
# 
# #### Searching for best $k$
# 
# Let us run GridSearch on both PCA components and K, to see if we can get a better model.

# In[ ]:


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
best_k_with_pca = estimator.best_params_['kneighborsclassifier__n_neighbors']
print("Best no. of PCA components: %d, neighbors: %d" % 
      (best_pca_components,
       best_k_with_pca))


# We find that PCA with 8 components, followed by KNN with 7 neighbors is the best combination.

# In[ ]:


# Do k-fold cross-validation, collecting both "test" accuracy and F1 
pipeline = make_pipeline(StandardScaler(), 
                         PCA(random_state=207, n_components=best_pca_components),
                         KNeighborsClassifier(n_neighbors=best_k_with_pca))

cv_scores = cross_validate(pipeline, perf_train_data_nonull, train_labels, cv=k_folds, scoring=['accuracy', 'f1'])
util.print_cv_results(cv_scores)


# With this combination, we get accuracy of 84% and F1 score of 0.63.

# Let us summarize our three runs of KNN so far.
# 
# ***
# 
# ### Summary
# 
# Model | CV Accuracy | (95% CI) | CV F1 | (95% CI)
# :---|:---:|:---:|:---:|:---:
# K-Nearest Neighbors (Most features) | 0.849 | (0.721, 0.976) | 0.637 | (0.328, 0.946)
# K-Nearest Neighbors (Top n Features) | 0.841 | (0.756, 0.926) | 0.586 | (0.376, 0.795)
# K-Nearest Neighbors (PCA, most features) | 0.841 | (0.714, 0.967) | 0.630 | (0.357, 0.903)
# 
# The first model gives the best F1.  We will now use it to run on the test set and also for the final table.

# In[ ]:


pipeline = make_pipeline(StandardScaler(), 
                         KNeighborsClassifier(n_neighbors=best_k_all_features))
pipeline.fit(perf_train_data_nonull, train_labels)
predicted_labels = pipeline.predict(perf_test_data_nonull)
knn_score_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
knn_score_f1 = metrics.f1_score(test_labels, predicted_labels)

print("On the test set, the model has an accuracy of {:.2f}% and an F1 score of {:.2f}."
     .format(knn_score_accuracy*100, knn_score_f1))


# On test set, we get 87% accuracy, and good F1 score at 0.73.

# We will use the last model run above (PCA, most features) to generate the school list.
# 
# ***
# 
# ## Recommendations for PASSNYC
# 
# Lastly, according to the methodology described in our [overview notebook](final_project_overview.ipynb), we will make our recommendations to PASSNYC based on an analysis of schools that the models show to have the highest opportunity to engage with Black and Hispanic students, in order to increase SHSAT registration in this population. We consider these to be the schools that are most likely to benefit from PASSNYC's intervention and engagement.

# In[ ]:


pipeline = make_pipeline(StandardScaler(),
                         KNeighborsClassifier(n_neighbors=best_k_all_features))

fp_df = util.run_model_get_ordered_predictions(pipeline, train_data, test_data,
                                               perf_train_data_nonull, perf_test_data_nonull,
                                               train_labels, test_labels)


# Now that we have the false positives, we will obtain a ranking of the schools that we can provide to PASSNYC.

# In[ ]:


df_passnyc = util.create_passnyc_list(fp_df, train_data, test_data,
                                 train_labels, test_labels)
# Write to CSV
df_passnyc.to_csv('results/results.knn.csv')

df_passnyc

