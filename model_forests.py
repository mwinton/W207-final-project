
# coding: utf-8

# # Random Forests Notebook
# [Return to project overview](final_project_overview.ipynb),
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# ### Importing Libraries and setting options
# 
# First we import necessary libraries, including our util functions, and set Pandas and Matplotlib options.

# In[19]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from util import our_train_test_split, read_data, get_dummies, print_cv_results
import pickle

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading our data using util cleanup and imputing
# 
# Our util module has shared utility functions for cleaning up our data and imputing means.

# In[2]:


# Read the cleaned, merged, and mean-imputed data from our utility function
train_data, test_data, train_labels, test_labels = read_data(do_imputation=True)


# We'll drop some columns that were used to calculate our dependendt variable, as well as our index column, school name strings, and `school_income_estimate` which had too many missing values to fill via imputing.

# In[15]:


# We drop a few features for the following reasons:
#    Used in generating dependent variable: 'num_shsat_test_takers',
#        'offers_per_student', 'pct_test_takers'
#    Strings or other non-features: 'dbn', 'school_name'
#    Too many empty values: 'school_income_estimate'
#    Data preserved in other features: 'zip', 'rigorous_instruction_rating',
#       'collaborative_teachers_rating', 'supportive_environment_rating',
#       'effective_school_leadership_rating',
#       'strong_family_community_ties_rating', 'trust_rating'
#    Found not to help model: 'district' (or one-hot encoding)

FEATURES_TO_DROP = ['dbn', 'school_name', 'zip', 'num_shsat_test_takers',
                    'offers_per_student', 'pct_test_takers', 'school_income_estimate',
                    'rigorous_instruction_rating','collaborative_teachers_rating',
                    'supportive_environment_rating',
                    'effective_school_leadership_rating',
                    'strong_family_community_ties_rating', 'trust_rating',
                    'district']

# We'll go ahead and drop total_columns_to_drop columns.
train_prepped = train_data.drop(FEATURES_TO_DROP,axis=1)
test_prepped = test_data.drop(FEATURES_TO_DROP,axis=1)


# In[14]:


# We confirm our resulting data has no more NAs
print("Confirm total of remaining NAs is: ",np.sum(np.sum(train_dropped.isna())))


# ### Optimizing a Random Forest Model on Cross-Validation
# 
# We now move into training our random forest model. To optimize our hyperparameter of how many trees to include in our forest, we use GridSearchCV and take advantage of its cross validation capability to use cross validation against our training set instead of further reducing our data into smaller train and dev sets. 

# In[8]:


# We check for previously saved results and a serialized model saved
# to disk before re-running GridSearchCV. To force it to run again,
# we can comment out the try: & except: or just delete the last saved
# results.

try:

    cv_results = pd.read_csv('cache_forest/forest_gridsearch_results.csv')
    with open('cache_forest/pickled_forest','rb') as f:
        forest_cv = pickle.load(f)

except:

    # If no saved results are found, we define our base Random Forest
    # Classifier with fixed parameters we don't anticipate adjusting.
    # We want to run as many jobs as we have cores at once (which is
    # what the -1 input to n_jobs does, and we define our random state
    # for reproducibility.)
    forest = RandomForestClassifier(n_jobs=-1, class_weight='balanced',
                                    random_state=207)

    # We define a range of paramters we'd like to try for our forest.
    params_to_try = {'n_estimators':[10,30,100,300],
                     'max_depth':[None,2,5,7],
                     'min_samples_leaf':[1,2,4],
                     'min_samples_split':[2,3,5],
                     'max_features':[.2,.5,.8]}

    # We define into how many groups we'd like to split our test data
    # for use in cross-validation to evaluate hyperparameters.
    KFOLDS = 5

    # Now we run GridSearchCV on our forest estimator, trying our varying
    # numbers of trees and utilizing our cross validation to determine the
    # best number of trees across the best number of train/dev cross
    # validation splits, using a weighted F1 score as our metric of success.
    forest_cv = GridSearchCV(forest, params_to_try, scoring=['f1',
                            'accuracy'], refit='f1', cv=KFOLDS,
                             return_train_score=False)

    # We'll time it and report how long it took to run:
    start_time = time.time()
    forest_cv.fit(train_prepped, train_labels)
    end_time = time.time()
    
    took = int(end_time - start_time)
    print("Grid search took {0:d} minutes, {1:d} seconds.".format(
              took // 60, took % 60))

    # And pickle our trained model, and save our scores to csv.
    with open('cache_forest/pickled_forest','wb') as f:
        pickle.dump(forest_cv, f)

    cv_results = pd.DataFrame(forest_cv.cv_results_)
    cv_results.to_csv('cache_forest/forest_gridsearch_results.csv')
    
# Then display our results in a Pandas dataframe, sorted by
# rank based on mean f1 score across 5-fold CV testing:
cv_results.sort_values('rank_test_f1')


# In[10]:


# We extract our best model and best parameters from our GridSearchCV results.
best_forest = forest_cv.best_estimator_
best_params = forest_cv.best_params_

print("Best params:\n")
for param, val in best_params.items():
    print(param,':',val)

print("\n")

winning_cv_results = cv_results[cv_results['rank_test_f1'] == 1].iloc[1,:]

# display accuracy with 95% confidence interval
winning_mean_accuracy = winning_cv_results['mean_test_accuracy']
std_accuracy = winning_cv_results['std_test_accuracy']
print('With %d-fold cross-validation,\nAccuracy is: %.3f (95%% CI from %.3f to %.3f).' %
          (KFOLDS, winning_mean_accuracy,
           float(winning_mean_accuracy - 1.96 * std_accuracy),
           float(winning_mean_accuracy + 1.96 * std_accuracy)))

# display F1 score with 95% confidence interval
winning_mean_f1 = winning_cv_results['mean_test_f1']
std_f1 = winning_cv_results['std_test_f1']
print('The F1 score is: %.3f (95%% CI from %.3f to %.3f).' %
          (winning_mean_f1,
           float(winning_mean_f1 - 1.96 * std_f1),
           float(winning_mean_f1 + 1.96 * std_f1)))


# ### Analyzing our Top 10 Features
# These features have the highest feature importance scores as found by our best forest model.
# 
# Unsurprisingly, they tend to include our most general metrics of performance, like average proficiency and grade 7 ela scores of 4 across all students.

# In[16]:


features = train_prepped.columns
feature_importances = best_forest.feature_importances_

features_and_importances = pd.DataFrame(feature_importances,features,['Importances'])

# Need column names here after the ohe_data step to analyze the results
features_and_importances.sort_values('Importances', ascending=False).iloc[1:11,]


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


# ### Recommendations Based on False Positives
# 
# We will make our recommendations based on our false positives (i.e. schools that our model
# thinks should be ranked as 'high_registrations', but for whatever reason, aren't).

# In[20]:


# recombine train and test data into an aggregate dataset
X_orig = pd.concat([train_data, test_data], sort=True)  # including all columns (need for display purposes)
X_best = pd.concat([train_prepped, test_prepped], sort=True)  # only columns from what ended up in my best model
y = np.concatenate((train_labels,test_labels))

X_best_npa = np.array(X_best)
y_npa = np.array(y)
X_pos = X_best[y==1]
X_neg = X_best[y==0]

# Run k-fold cross-validation with 5 folds 10 times, which means every school is predicted 10 times.
folds = 5
repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=207)

# Build dataframes for storing predictions, with columns for each k-fold
fold_list = []
for f in range(1, (folds * repeats) + 1):
    fold_list.append('k{}'.format(f))
predictions = pd.DataFrame(index=X_best.index, columns=fold_list)


# In[21]:


# Iterate through the Repeated Stratified K Fold, and and fill out the DataFrames
counter = 1
print('Please be patient...')
for train, test in rskf.split(X_best_npa, y_npa):
    # TODO: it might be possible to refactor this into util if caller passes
    # in a configured pipeline
    best_forest.fit(X_best_npa[train], y_npa[train])
    predicted_labels = best_forest.predict(X_best_npa[test])
    predictions.iloc[test, counter-1] = predicted_labels
    counter += 1


# In[22]:


predictions.head()


# In[23]:


# Create columns for predictions and labels
predictions['1s'] = predictions.iloc[:,:50].sum(axis=1)
predictions['0s'] = (predictions.iloc[:,:50]==0).sum(axis=1)
predictions['true'] = y

# Create a table of raw results, the vote for truth
X_predicted = pd.concat([X_best, predictions['1s'], predictions['0s'],
                         pd.DataFrame(y)], axis=1, join_axes=[X_best.index])
X_predicted = X_predicted.sort_values(by=['1s', '0s'], ascending=[False, True])

# list all false positives that had at least 5/50 votes for the positive label
true_negatives = predictions[predictions['true']==0]
false_positives = true_negatives[true_negatives['1s'] > 5].sort_values(by='1s', ascending=False)['1s']

# join back to full dataset for all columns (including those previously dropped)
fp_result = pd.concat([false_positives, X_orig], axis=1, join='inner')
fp_result


# In[24]:


fp_result.info()


# In[31]:


# Retrain only the columns of interest for PASSNYC prioritization
fp_features = ['1s',
              'dbn',
              'school_name',
              'economic_need_index',
              'grade_7_enrollment',
              'num_shsat_test_takers',
              'pct_test_takers',
              'percent_black__hispanic'
              ]
df_passnyc = fp_result.loc[:,fp_features]

# Determine the number of test takers this school would have needed to meet
#   the median percentage of high_registrations
median_pct = np.median(X_orig[y==1]['pct_test_takers'])/100
target_test_takers = np.multiply(df_passnyc['grade_7_enrollment'], median_pct)

# Subtract the number of actual test takers from the hypothetical minimum number
delta = target_test_takers - df_passnyc['num_shsat_test_takers']

# Multiply the delta by the minority percentage of the school to determine how many minority
#   students did not take the test
minority_delta = np.multiply(delta, df_passnyc['percent_black__hispanic']/100).astype(int)

# Add this number to the dataframe, sort descending, and filter to schools with more than five minority students
df_passnyc['minority_delta'] = minority_delta
df_passnyc = df_passnyc[df_passnyc['minority_delta'] > 5].sort_values(by='minority_delta', ascending=False)

# Create a rank order column
df_passnyc.insert(0, 'rank', range(1,df_passnyc.shape[0]+1))

# Write to CSV
df_passnyc.to_csv('results/results.randomforest.csv')

df_passnyc

