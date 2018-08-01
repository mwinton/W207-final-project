
# coding: utf-8

# # Random Forests Notebook
# [Return to project overview](final_project_overview.ipynb),
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# ### Importing Libraries and setting options
# 
# First we import necessary libraries, including our util functions, and set Pandas and Matplotlib options.

# In[ ]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from util import our_train_test_split, read_data, ohe_data, print_cv_results
import pickle

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading our data using util cleanup and imputing
# 
# Our util module has shared utility functions for cleaning up our data and imputing means.

# In[ ]:


# Read the cleaned, merged, and mean-imputed data from our utility function
train_data, test_data, train_labels, test_labels = read_data(do_imputation=True)


# We'll drop some columns that were used to calculate our dependendt variable, as well as our index column, school name strings, and `school_income_estimate` which had too many missing values to fill via imputing.

# In[ ]:


# We drop a few features for the following reasons:
#    Used in generating dependent variable: 'num_shsat_test_takers',
#        'offers_per_student', 'pct_test_takers'
#    Strings or other non-features: 'dbn', 'school_name'
#    Too many empty values: 'school_income_estimate'
#    Data preserved in other features (namely the 'in_[borough]' derived features): 'zip'

FEATURES_TO_DROP = ['dbn', 'school_name', 'zip', 'num_shsat_test_takers',
                    'offers_per_student', 'pct_test_takers', 'school_income_estimate']

# We'll go ahead and drop total_columns_to_drop columns.
train_dropped = train_data.drop(FEATURES_TO_DROP,axis=1)
test_dropped = test_data.drop(FEATURES_TO_DROP,axis=1)


# In[ ]:


# We confirm our resulting data has no more NAs
print("Confirm total of remaining NAs is: ",np.sum(np.sum(train_dropped.isna())))


# ### One-Hot Encoding of Districts
# 
# We have already binarized our boroughs, but let's do the same to our districts via our "one hot encoding" utility function.

# In[ ]:


train_prepped, test_prepped = ohe_data(train_dropped, test_dropped, ['district'])


# ### Optimizing a Random Forest Model on Cross-Validation
# 
# We now move into training our random forest model. To optimize our hyperparameter of how many trees to include in our forest, we use GridSearchCV and take advantage of its cross validation capability to use cross validation against our training set instead of further reducing our data into smaller train and dev sets. 

# In[ ]:


# NOTE: This cell takes a while to run, so if you want to skip it,
# simply run the cell below to load the csv report of the most recent run.

# First we define our base Random Forest Classifier with fixed parameters we don't
# anticipate adjusting. We want to run as many jobs as we have cores at once
# (which is what the -1 input to n_jobs does, and we define our random state
# for reproducibility.)
forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced',random_state=207)

# We define a range of paramters we'd like to try for our forest.
params_to_try = {'n_estimators':[10,30,100,300], 'max_depth':[None,2,5,7],
                 'min_samples_leaf':[1,2,4], 'min_samples_split':[2,3,5],
                 'max_features':[.2,.5,.8]}

# We define into how many groups we'd like to split our test data
# for use in cross-validation to evaluate hyperparameters.
KFOLDS = 5

# Now we run GridSearchCV on our forest estimator, trying our varying numbers of trees
# and utilizing our cross validation to determine the best number of trees across the
# best number of train/dev cross validation splits, using a weighted F1 score as our
# metric of success.
forest_cv = GridSearchCV(forest, params_to_try, scoring='f1_weighted', cv=KFOLDS,
                        return_train_score=False)
forest_cv.fit(train_prepped, train_labels)

with open('cache_forest/pickled_forest','wb') as f:
    pickle.dump(forest_cv, f)

cv_results = pd.DataFrame(forest_cv.cv_results_)
cv_results.to_csv('cache_forest/forest_gridsearch_results.csv')


# In[ ]:


# Skip to here to read the results
# and reload our serialized GridSearchCV with our
# best estimator and parameters.

cv_results = pd.read_csv('cache_forest/forest_gridsearch_results.csv')

with open('cache_forest/pickled_forest','rb') as f:
    forest_cv = pickle.load(f)

# And we display the results in a Pandas dataframe.
cv_results


# In[ ]:


# We extract our best model and best parameters from our GridSearchCV results.
best_forest = forest_cv.best_estimator_
best_params = forest_cv.best_params_

# As well as the results of the best parameters on the CV
winning_cv_results = cv_results[cv_results['rank_test_score'] == 1]

print("Best params:\n")
for param, val in best_params.items():
    print(param,':',val)

# The winning results in a Dataframe
winning_cv_results

# I'll need to adapt or recreate the results of the 
# util.print_cv_results method
# print_cv_results(winning_cv_results)


# ### Maximizing our maximum mean score across cross validations
# 
# Before moving on testing our model against our test data, we have adjusted our features, preprocessing, and hyperparameters to maximize the highest mean test score we receive.

# In[ ]:


print("Maximum mean score across cross validations: {0:.4f}"      .format(np.max(cv_results['mean_test_score'])))


# ### Analyzing our most important features

# In[ ]:


# Need column names here after the ohe_data step to analyze the results
print(best_forest.feature_importances_)


# ### Measuring results on the test set
# 
# Now that we have determined our best preprocessing steps and hyperparameters,
# we evaluate our results on our test set.

# In[ ]:


# We train on our full training data on a new forest with our best_params
# determined by our GridSearchCV
# best_forest.fit(train_prepped, train_labels)

# And make predictions on our test data
# predictions = best_forest.predict(test_prepped)
# f1 = f1_score(test_labels, predictions, average='weighted')
# f1 = f1_score(test_labels, predictions)
# accuracy = np.sum(predictions == test_labels) / len(test_labels)
    
# print("Weighted Average F1 Score: {0:.4f}".format(f1))
# print("Accuracy: {0:.4f}".format(accuracy))

