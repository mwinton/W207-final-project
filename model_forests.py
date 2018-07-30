
# coding: utf-8

# # Random Forests Notebook
# [Return to project overview](final_project_overview.ipynb),
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# ### Importing Libraries and setting options
# 
# First we import necessary libraries, including our util functions, and set Pandas and Matplotlib options.

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from util import our_train_test_split, read_data, ohe_data, print_cv_results

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading our data using util cleanup and imputing
# 
# Our util module has shared utility functions for cleaning up our data and imputing means.

# In[2]:


# Read the cleaned, merged, and mean-imputed data from our utility function
train_data, test_data, train_labels, test_labels = read_data()


# We'll drop some columns that were used to calculate our dependendt variable, as well as our index column, school name strings, and `school_income_estimate` which had too many missing values to fill via imputing.

# In[3]:


# We know we need to drop some variables due to them being an irrelevant index ('Unnamed: 0'),
# or being a string, or related to calculating our dependent variable.
FEATURES_TO_DROP = ['dbn', 'school_name',                     'num_shsat_test_takers', 'offers_per_student', 'pct_test_takers',
                   'school_income_estimate']

# We also know that we still need to clean up some NA values. First we'll look for
# columns with too many NA values, and add those to our list of FEATURES TO DROP.
bool_of_too_many_missing_values = train_data.isna().sum() >= 10
more_columns_to_drop = train_data.columns[bool_of_too_many_missing_values]
total_columns_to_drop = FEATURES_TO_DROP + list(more_columns_to_drop)

# It may be that the ability of a school to calculate a school_income_estimate
# is related to its test score ability, so while we cannot use the school_income_estimate
# values, we can use the presence or absence of those values.
train_data['sie_provided'] = -train_data['school_income_estimate'].isna()
test_data['sie_provided'] = -test_data['school_income_estimate'].isna()

# We'll go ahead and drop total_columns_to_drop columns.
train_nona = train_data.drop(total_columns_to_drop,axis=1)
test_nona = test_data.drop(total_columns_to_drop,axis=1)


# In[4]:


# We confirm our resulting data has no more NAs
print("Confirm total of remaining NAs is: ",np.sum(np.sum(train_nona.isna())))


# ### One-Hot Encoding of Categorical Features
# 
# While zip codes might be too granular on their own to find commonalities between schools, we see if grouping by burough provides more shared information, and allow the district variable to stand in for more detailed locality.

# In[5]:


bronx_zips = [10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473]
brooklyn_zips = [11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237]
manhattan_zips = [10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040]
queens_zips = [11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378]
staten_zips = [10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312,10301, 10304, 10305,10314]

train_cat_zip = train_nona
train_cat_zip['in_bronx'] = train_nona['zip'].apply((lambda zip: zip in bronx_zips))
train_cat_zip['in_brooklyn'] = train_nona['zip'].apply((lambda zip: zip in brooklyn_zips))
train_cat_zip['in_manhattan'] = train_nona['zip'].apply((lambda zip: zip in manhattan_zips))
train_cat_zip['in_queens'] = train_nona['zip'].apply((lambda zip: zip in queens_zips))
train_cat_zip['in_staten'] = train_nona['zip'].apply((lambda zip: zip in staten_zips))

test_cat_zip = test_nona
test_cat_zip['in_bronx'] = test_nona['zip'].apply((lambda zip: zip in bronx_zips))
test_cat_zip['in_brooklyn'] = test_nona['zip'].apply((lambda zip: zip in brooklyn_zips))
test_cat_zip['in_manhattan'] = test_nona['zip'].apply((lambda zip: zip in manhattan_zips))
test_cat_zip['in_queens'] = test_nona['zip'].apply((lambda zip: zip in queens_zips))
test_cat_zip['in_staten'] = test_nona['zip'].apply((lambda zip: zip in staten_zips))

train_borough_ohe = train_cat_zip.drop(['zip'],axis=1)
test_borough_ohe = test_cat_zip.drop(['zip'],axis=1)

print("Train Borough OHE shape:", train_borough_ohe.shape)
print("Test Borough OHE shape:", test_borough_ohe.shape)


# To ensure that our district categorical variable is not interpreted as an integer value, we binarize it via our "one hot encoding" utility function.

# In[6]:


# Here we use our ohe_data util function to create binarized columns of our
# categorical district data.

OHE_COLS = ['district']
train_ohe, test_ohe = ohe_data(train_borough_ohe, test_borough_ohe, OHE_COLS)


# In[7]:


# Because our ohe_data function returns a sparse matrix, we fill NAs with
# 0s to return to a dense matrix
train_prepped = pd.SparseDataFrame(train_ohe, default_fill_value=0)
test_prepped = pd.SparseDataFrame(test_ohe, default_fill_value=0)

print("Train prepped shape:", train_prepped.shape)
print("Test prepped shape:", test_prepped.shape)


# ### Optimizing a Random Forest Model on Cross-Validation
# 
# We now move into training our random forest model. To optimize our hyperparameter of how many trees to include in our forest, we use GridSearchCV and take advantage of its cross validation capability to use cross validation against our training set instead of further reducing our data into smaller train and dev sets. 

# In[32]:


# First we define our base Random Forest Classifier with fixed parameters we don't
# anticipate adjusting. We want no limit to the features used in each tree, we
# want to run as many jobs as we have cores at once (which is what the -1 input
# to n_jobs does, and we define our random state for reproducibility.)
forest = RandomForestClassifier(max_features=None, n_jobs=-1, random_state=207)

# We define a range of numbers of trees we'd like to include
# in our forests for input into our GridSearch.
params_to_try = {'n_estimators':[1,3,10,30,100,300]}

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


# In[33]:


# We extract our best model and best parameters from our GridSearchCV results.
best_forest = forest_cv.best_estimator_
best_n_estimators = forest_cv.best_params_['n_estimators']
print("Best no. estimators: ", best_n_estimators)

# And we display the results in a Pandas dataframe.
cv_results = pd.DataFrame(forest_cv.cv_results_)
cv_results


# ### Maximizing our maximum mean test score
# 
# Before moving on testing our model against our test data, we have adjusted our features, preprocessing, and hyperparameters to maximize the highest mean test score we receive.

# In[34]:


print("Max mean test score: {0:.4f}".format(np.max(cv_results['mean_test_score'])))


# ### Measuring results on the test set
# 
# Now that we have determined our best preprocessing steps and hyperparameters,
# we evaluate our results on our test set.

# In[35]:


# We train on our full training data on our best_forest determined by our GridSearchCV
best_forest.fit(train_prepped, train_labels)

# And make predictions on our test data
predictions = best_forest.predict(test_prepped)
f1 = f1_score(test_labels, predictions, average='weighted')
accuracy = np.sum(predictions == test_labels) / len(test_labels)
    
print("\nTrees in our Forest:",best_n_estimators)
print("Weighted Average F1 Score: {0:.4f}".format(f1))
print("Accuracy: {0:.4f}".format(accuracy))


# We see that with 100 trees in our random forest, we achieve a Weighted Average F1 Score of 0.8730, and an accuracy of 87.1%.
