
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


# In[17]:


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

get_ipython().run_line_magic('matplotlib', 'inline')



# In[27]:



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


# In[20]:


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
perf_train_data.info()
perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())


# ### K-Nearest Neighbors Classification
# 
# We will now run KNN prediction on the dataset, with the default K value (=5).

# In[21]:


scaler = MinMaxScaler().fit(perf_train_data_nonull)
rescaledX = scaler.transform(perf_train_data_nonull)
y = train_labels
clf = KNeighborsClassifier()

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds = 5
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# We get accuracy of 82% and F1 score of 0.58.  Let us experiment with various values of $k$ to see which gives the best results.


# In[22]:


pipeline = make_pipeline(MinMaxScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=5, n_jobs=2, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

print("Best no. of neighbors: %d (with best f1: %.3f)" % 
      (estimator.best_params_['kneighborsclassifier__n_neighbors'], 
       estimator.best_score_))


# The best F1 score is 0.61 at $k=13$.

# ### KNN with select features
# 
# We will now attempt to do some feature selection, followed by running KNN.


# In[23]:


pipeline = make_pipeline(MinMaxScaler(), 
                         SelectFromModel(ExtraTreesClassifier(random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, y)
selected_features = pipeline.steps[1][1].get_support()
perf_train_data_nonull.columns[selected_features]



# In[24]:


perf_train_data_nonull_sel_cols = ['economic_need_index', 'percent_asian', 'percent_hispanic',
       'percent_black__hispanic', 'student_attendance_rate',
       'percent_of_students_chronically_absent',
       'supportive_environment_percent', 'student_achievement_rating',
       'average_ela_proficiency', 'average_math_proficiency',
       'grade_7_ela_4s_all_students', 'grade_7_ela_4s_hispanic_or_latino',
       'grade_7_math_4s_all_students',
       'grade_7_math_4s_black_or_african_american',
       'grade_7_math_4s_asian_or_pacific_islander',
       'grade_7_math_4s_economically_disadvantaged',
       'number_of_students_social_studies', 'average_class_size_english',
       'average_class_size_science', 'school_pupil_teacher_ratio']
perf_train_data_nonull_sel = perf_train_data_nonull[perf_train_data_nonull_sel_cols]
scaler = MinMaxScaler().fit(perf_train_data_nonull_sel)
rescaledX = scaler.transform(perf_train_data_nonull_sel)
y = train_labels
clf = KNeighborsClassifier(n_neighbors=13)

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds = 5
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# F1 score falls a tiny bit to 0.60.  We can ignore this set and use the original set instead.

# ### KNN with reduced dimensions
# 
# We will next attempt to reduce dimensions via PCA, followed by KNN.
# 
# First, we will attempt to find the best number of components.


# In[25]:


# generate plot of variance explained vs # principale components
util.get_num_pcas(perf_train_data_nonull, var_explained=0.9)


# We can see that the first 3 components already explain more than 70% of variance.  The slope of the graph goes down after this, indicating that remaining components are not as informative.
# 
# Let us run GridSearch on both PCA components and K, to see if we can get a better model.


# In[26]:


pipeline = make_pipeline(StandardScaler(), 
                         PCA(random_state=207),
                         KNeighborsClassifier())

n_components = list(range(1, 12))
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(pca__n_components=n_components,
                             kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=5, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

print("Best no. of PCA components: %d, neighbors: %d (with best f1: %.3f)" % 
      (estimator.best_params_['pca__n_components'],
       estimator.best_params_['kneighborsclassifier__n_neighbors'], 
       estimator.best_score_))


# PCA with 9 components, followed by KNN with 13 neighbors, gives us F1-score that's up by 0.005: earlier, it was 0.614, now it's 0.619.  But we also lose a lot of interpretability; it may not be worth it to go down this path.
