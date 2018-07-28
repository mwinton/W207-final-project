
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import PCA

import util

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

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
# We ignore the following demographic indicators:
# * school_name
# * zip
# * district
# * community_school
# * economic_need_index
# * school_income_estimate
# * percent_ell
# * percent_black
# * percent_hispanic
# * percent_asian
# * percent_white
# * percent_of_students_chronically_absent
# 
# We also ignore the following columns because they proxy output variable:
# * num_shsat_test_takers
# * offers_per_student
# * pct_test_takers

# In[3]:


# To generate this list again:
# Take above (markdown) list and store it in say ~/tmp/col_list.  Then:
# cat  ~/tmp/col_list | cut -d" " -f2 | sed -E 's/^(.*)$/"\1"/' | tr '\n' ', '

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
    'community_school',
    'economic_need_index',
    'school_income_estimate',
    'percent_ell',
    'percent_black',
    'percent_black__hispanic',
    'percent_hispanic',
    'percent_asian',
    'percent_white',
    'grade_7_ela_4s_american_indian_or_alaska_native',
    'grade_7_ela_4s_black_or_african_american',
    'grade_7_ela_4s_hispanic_or_latino',
    'grade_7_ela_4s_asian_or_pacific_islander',
    'grade_7_ela_4s_white',
    'grade_7_ela_4s_multiracial',
    'grade_7_ela_4s_limited_english_proficient',
    'grade_7_ela_4s_economically_disadvantaged',
    'grade_7_math_4s_american_indian_or_alaska_native',
    'grade_7_math_4s_black_or_african_american',
    'grade_7_math_4s_hispanic_or_latino',
    'grade_7_math_4s_asian_or_pacific_islander',
    'grade_7_math_4s_white',
    'grade_7_math_4s_multiracial',
    'grade_7_math_4s_limited_english_proficient',
    'grade_7_math_4s_economically_disadvantaged',
]
perf_train_data = train_data.drop(drop_cols, axis=1)
perf_train_data.info()
perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())


# ### K-Nearest Neighbors Classification
# 
# We will now run KNN prediction on the dataset, with the default K value (=5).

# In[4]:


scaler = MinMaxScaler().fit(perf_train_data_nonull)
rescaledX = scaler.transform(perf_train_data_nonull)
y = train_labels.values.ravel()
clf = KNeighborsClassifier()

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds = 10
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# We get accuracy of 83% and F1 score of 0.58.  Let us experiment with various values of $k$ to see which gives the best results.

# In[5]:


pipeline = make_pipeline(MinMaxScaler(), 
                         KNeighborsClassifier())
n_neighbors = list(range(1, 15))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=10, n_jobs=2, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

print("Best no. of neighbors: %d (with best f1: %.3f)" % 
      (estimator.best_params_['kneighborsclassifier__n_neighbors'], 
       estimator.best_score_))


# The best F1 score is 0.62 at $k=3$.

# ### KNN with select features
# 
# We will now attempt to do some feature selection, followed by running KNN.

# In[6]:


pipeline = make_pipeline(MinMaxScaler(), 
                         SelectFromModel(ExtraTreesClassifier(random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, y)
selected_features = pipeline.steps[1][1].get_support()
perf_train_data_nonull.columns[selected_features]


# In[7]:


perf_train_data_nonull_sel_cols = ['student_attendance_rate', 'percent_of_students_chronically_absent',
       'student_achievement_rating', 'average_ela_proficiency',
       'average_math_proficiency', 'grade_7_math_4s_all_students',
       'number_of_students_social_studies', 'average_class_size_science']
perf_train_data_nonull_sel = perf_train_data_nonull[perf_train_data_nonull_sel_cols]
scaler = MinMaxScaler().fit(perf_train_data_nonull_sel)
rescaledX = scaler.transform(perf_train_data_nonull_sel)
y = train_labels.values.ravel()
clf = KNeighborsClassifier(n_neighbors=3)

# Do k-fold cross-validation, collecting both "test" accuracy and F1 
k_folds = 10
cv_scores = cross_validate(clf, rescaledX, y, cv=k_folds, scoring=['accuracy','f1'])
util.print_cv_results(cv_scores)


# F1 score falls by about 4%.  We can ignore this set and use the original set instead.

# ### KNN with reduced dimensions
# 
# We will next attempt to reduce dimensions via PCA, followed by KNN.
# 
# First, we will attempt to find the best number of components.

# In[8]:


cum_explained_variance_ratios = []
for n in range(1, 15):
    pipeline = make_pipeline(StandardScaler(), 
                            PCA(n_components=n, random_state=207))
    pipeline.fit_transform(perf_train_data_nonull)
    pca = pipeline.steps[1][1]
    cum_explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

import seaborn as sns
sns.set()
plt.plot(np.array(cum_explained_variance_ratios))
plt.show()


# We will select first 3 components, which already explain more than 70% of variance.  The slope of the graph goes down after this, indicating that remaining components are not as informative.

# In[9]:


pipeline = make_pipeline(StandardScaler(), 
                         PCA(n_components=3, random_state=207),
                         KNeighborsClassifier())

n_neighbors = list(range(1, 10))
estimator = GridSearchCV(pipeline,
                        dict(kneighborsclassifier__n_neighbors=n_neighbors),
                        cv=10, scoring='f1')
estimator.fit(perf_train_data_nonull, y)

print("Best no. of neighbors: %d (with best f1: %.3f)" % 
      (estimator.best_params_['kneighborsclassifier__n_neighbors'], 
       estimator.best_score_))


# We notice that F1 score now goes up a tad, to 0.65.  We lose a lot of interpretability; it may not be worth to go down .
