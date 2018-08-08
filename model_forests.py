
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
import time
import graphviz

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from util import our_train_test_split, read_data, get_dummies,     print_cv_results, run_model_get_ordered_predictions, create_passnyc_list
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

# In[3]:


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


# In[4]:


# We confirm our resulting data has no more NAs
print("Confirm total of remaining NAs is: ",np.sum(np.sum(train_prepped.isna())))


# ### Optimizing a Random Forest Model on Cross-Validation
# 
# We now move into training our random forest model. To optimize our hyperparameter of how many trees to include in our forest, we use GridSearchCV and take advantage of its cross validation capability to use cross validation against our training set instead of further reducing our data into smaller train and dev sets. 

# In[5]:


# We define into how many groups we'd like to split our test data
# for use in cross-validation to evaluate hyperparameters.
KFOLDS = 5

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


# In[6]:


# We extract our best model and best parameters from our GridSearchCV results.
best_forest = forest_cv.best_estimator_
best_params = forest_cv.best_params_

# We reiterate our preferred number of cross validation folds if we haven't
# had to re-train our model
KFOLDS = 5

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
# 
# It is interesting to see how heavily the absence and attendance rates factor in.

# In[7]:


features = train_prepped.columns
feature_importances = best_forest.feature_importances_

features_and_importances = pd.DataFrame(feature_importances,features,['Importances'])

# Need column names here after the ohe_data step to analyze the results
features_and_importances.sort_values('Importances', ascending=False).iloc[1:11,]


# ### Viewing a few trees from our forest
# 
# To see what decisions some of our trees are coming to, let's take a look at three random trees out of our group of estimators.

# In[10]:


trees_in_forest = best_forest.estimators_
max_index = len(trees_in_forest)

random_indeces = np.random.randint(0, max_index, 3)
example_graphs = []

for index in random_indeces:
    tree_viz = export_graphviz(trees_in_forest[index], proportion=True, filled=True,
                               feature_names=train_prepped.columns, rounded=True,
                               class_names=['not_high_registrations','high_registrations'],
                               out_file=None)
    try:
        graphviz.Source(source=tree_viz,
                        filename='cache_forest/tree_viz_{0}'.format(index),
                        format='svg').render()
    except ExecutableNotFound:
        print("Your system lacks GraphViz. Instructions to install for your" +             "operating system should be available at https://graphviz.gitlab.io/download/" +             "The images will be loaded and linked to below, so you don't need it to view" +             "this notebook.")


# In the displayed graphs below, the more orange a cell is, the more the samples that pass through it tend to be not in our "high_registrations" category. The more blue a cell is, the more it tends to include "high_registrations." We are using the gini measurement of impurity to structure our trees.
# 
# The samples percentage tells us what percentage of our total samples pass through this node.
# 
# The value list tells us how many of the samples that have reached this node are in each class. So the first value (value[0]) indicates what proportion of the samples in the node are not high_registrations, and the second value (value[1]) tells us how many are high_registrations. You can see that these values correspond to the coloring of the graph.
# 
# Then from each node, if a sample meets the condition that titles the node, it travels to the lower left branch. If it does not meed the condition of the node, it travels down the right branch.

# #### Graph of Tree #13
# 
# ![Graph #13](cache_forest/tree_viz_13.svg)
# 
# [Link to Graph #13 if not rendering on GitHub](https://www.dropbox.com/s/vqg9hm8ol2kxy7d/tree_viz_13.svg?dl=0)

# #### Graph of Tree #39
# 
# ![Graph #39](cache_forest/tree_viz_39.svg)
# 
# [Link to Graph #39 if not rendering on GitHub](https://www.dropbox.com/s/x0ny1fpk13yj16c/tree_viz_39.svg?dl=0)

# #### Graph of Tree #77
# 
# ![Graph #39](cache_forest/tree_viz_77.svg)
# 
# [Link to Graph #77 if not rendering on GitHub](https://www.dropbox.com/s/rlspal912qu0euf/tree_viz_77.svg?dl=0)

# Remember these are just three out of our total 100 trees that make our ensemble predictor, and each of these trees only have half of the total features in our set. Their variation is what helps 'smooth out the edges' of some of the predictions, to gain the benefits of an ensemble within a single model.
# 
# All in all, the graph results are to be expected given the features that we found to be important, but the PASSNYC team specifically asked for models that could be explained, and we feel these trees would of course help explain the model's decision-making process clearly to all stakeholders.

# ### Measuring results on the test set
# 
# Now that we have determined our best preprocessing steps and hyperparameters,
# we evaluate our results on our test set.

# In[11]:


# We train on our full training data on a new forest with our best_params
# determined by our GridSearchCV
best_forest.fit(train_prepped, train_labels)
predictions = best_forest.predict(test_prepped)

# And make predictions on our test data
# predictions = best_forest.predict(test_prepped)
f1 = f1_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
accuracy = np.sum(predictions == test_labels) / len(test_labels)
    
print("Average F1 Score: {0:.4f}".format(f1))
print("Accuracy: {0:.4f}".format(accuracy))


# ### Recommendations Based on False Positives
# 
# We will make our recommendations based on our false positives (i.e. schools that our model
# thinks should be ranked as 'high_registrations', but for whatever reason, aren't).

# In[12]:


fp_df = run_model_get_ordered_predictions(best_forest, train_data, test_data,
                                      train_prepped, test_prepped,
                                      train_labels, test_labels)


# We now use another util function to generate the list we'll feed to our final
# ensemble evaluation.

df_passnyc = create_passnyc_list(fp_df, train_data, test_data, train_labels, test_labels)
# Write to CSV
df_passnyc.to_csv('results/results.randomforest.csv')

df_passnyc

