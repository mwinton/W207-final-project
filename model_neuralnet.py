
# coding: utf-8

# # Neural Network Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# In[44]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import util

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().magic('matplotlib inline')


# ## Load data and split class labels into separate array
# 
# Our utility function reads the merged dataset, imputes the column mean for missing numeric values, and then performs a stratified train-test split.

# In[17]:


train_data, test_data, train_labels, test_labels = util.read_data(do_imputation=True)
print(train_data.shape)
print(train_labels.shape)


# > **KEY OBSERVATION**: a hypothetical model that is hard-coded to predict a `negative` result every time would be ~77% accurate.  So, we should not accept any machine-learned model with a lower accuracy than that.  This also suggests that F1 score is a better metric to assess our work since it incorporates both precision and recall.

# In[18]:


train_data.info()
train_data.head(10)


# > **KEY OBSERVATION**: the feature `school_income_estimate` only has non-null values for 104 of 371 records in the training data.  We should drop it from further analysis, as imputing its value for the non-null records isn't appropriate.

# ## Create a function to estimate MLP models and report results

# In[19]:


def estimate_mlp(train_data, train_labels, n_pca=None,
                 hidden_layers=None, k_folds=5, max_iter=1000, print_results=True):

    # if tuple describing hidden layer nodes isn't provided, set default
    if not hidden_layers:
        n_features = train_data.shape[1]
        hidden_layers = (n_features,n_features,n_features)
        
    # build pipelines, with or without PCA as appropriate
    if n_pca:
        # create a pipeline to run StandardScaler and MLP
        print('Estimating pipeline with PCA; hidden layers:',hidden_layers)
        pipeline = make_pipeline(StandardScaler(with_mean=False), 
                                 PCA(n_components=n_pca, random_state=207),
                                 MLPClassifier(hidden_layer_sizes=hidden_layers,
                                               max_iter=max_iter, random_state=207))
    else:
        # create a pipeline to run StandardScaler and MLP
        print('Estimating pipeline without PCA; hidden layers:',hidden_layers)
        pipeline = make_pipeline(StandardScaler(with_mean=False), 
                                 MLPClassifier(hidden_layer_sizes=hidden_layers,
                                               max_iter=max_iter, random_state=207))

    # Do k-fold cross-validation, collecting both "test" accuracy and F1 
    cv_scores = cross_validate(pipeline, train_data, train_labels, cv=k_folds, scoring=['accuracy','f1'])
    if print_results:
        util.print_cv_results(cv_scores)
        
    # extract and return accuracy, F1
    cv_accuracy = cv_scores['test_accuracy']
    cv_f1 = cv_scores['test_f1']
    return (cv_accuracy.mean(), cv_accuracy.std(), cv_f1.mean(), cv_f1.std())


# ## Train and fit a "naive" model
# For the first model, we'll use all features except SHSAT-related features because they are too correlated with the way we calculated the label.  We'll also drop `school_income_estimate` because it's missing for ~2/3 of the schools.  We drop zip code (too granular to have many schools per zip) in favor of the indicator variables `in_[borough]`.

# In[20]:


drop_cols = ['dbn',
             'num_shsat_test_takers',
             'offers_per_student',
             'pct_test_takers',
             'school_name',
             'school_income_estimate',
             'zip'
            ]

# drop SHSAT-related columns
train_data_naive = train_data.drop(drop_cols, axis=1)
test_data_naive = test_data.drop(drop_cols, axis=1)

print(train_data_naive.shape)
train_data_naive.head()


# ## One Hot Encode the categorical explanatory variables
# Columns such as zip code and school district ID, which are integers should not be fed into an ML model as integers.  Instead, we would need to treat them as factors and perform one-hot encoding.  Since we have already removed zip code from our dataframe (in favor of boroughs), we only need to one hot encode `district`.

# In[21]:


train_data_naive_ohe, test_data_naive_ohe = util.get_dummies(train_data_naive, test_data_naive,
                                                             factor_cols=['district'])
train_data_naive_ohe.head()


# ## Estimate the "naive" multilayer perceptron model
# This first "naive" model uses all except for the SHSAT-related features, as described above.  We create a pipeline that will be used for k-fold cross-validation.  First, we scale the features, then estimate a multilayer perceptron neural network with 3 hidden layers, each with the same number of nodes as we have features.

# In[22]:


# discard return vals; only print results
(_,_,_,_) = estimate_mlp(train_data_naive_ohe, train_labels, k_folds=5, max_iter=1000)


# ## Train a "naive" model without location (zip, borough, or district)
# Next, we will remove the borough and district features and compare accuracy to the model that included one hot-encoded versions of the borough and district factors.

# In[23]:


drop_cols = ['dbn',
             'num_shsat_test_takers',
             'offers_per_student',
             'pct_test_takers',
             'school_name',
             'school_income_estimate',
             'district',
             'zip',
             'in_bronx',
             'in_brooklyn',
             'in_manhattan',
             'in_queens',
             'in_staten'
            ]

# drop SHSAT-related columns + district, zip, borough
train_data_naive_nozip = train_data.drop(drop_cols, axis=1)
test_data_naive_nozip = test_data.drop(drop_cols, axis=1)

print(train_data_naive_nozip.shape)
print(train_labels.shape)


# ## Estimate the "naive" multilayer perceptron model without location
# 

# In[24]:


# discard return vals; only print results
(_,_,_,_) = estimate_mlp(train_data_naive_nozip, train_labels, k_folds=5, max_iter=1000)


# > **KEY OBSERVATION**: while the accuracy is similar when we exclude zip code and school district, the F1 score is lower.  This suggests that it's important to keep these factors in the model. 

# ## Train a "race-blind" multilayer perceptron model
# Because we know there's an existing bias problem in the NYC schools, in that the demographics of the test taking population have been getting more homogenous, and the explicit goal of PASSNYC is to make the pool more diverse, we want to train a model that excludes most demographic features.  This would enable us to train a "race-blind" model.  
# 
# ### Preprocess new X_train and X_test datasets
# We will remove all explicitly demographic columns, as well as economic factors, borough, and zip code, which are likely highly correlated with demographics.

# In[25]:


# drop SHSAT-related columns
drop_cols = ['dbn',
             'num_shsat_test_takers',
             'offers_per_student',
             'pct_test_takers',
             'school_name',
             'school_income_estimate'
            ]
train_data_race_blind = train_data.drop(drop_cols, axis=1)
test_data_race_blind = test_data.drop(drop_cols, axis=1)

# drop additional (demographic) columns
race_cols = ['percent_ell',
             'percent_asian',
             'percent_black',
             'percent_hispanic',
             'percent_black__hispanic',
             'percent_white',
             'economic_need_index',
             'zip',
             'in_bronx',
             'in_brooklyn',
             'in_manhattan',
             'in_queens',
             'in_staten'
             ]
train_data_race_blind = train_data_race_blind.drop(race_cols, axis=1)
test_data_race_blind = test_data_race_blind.drop(race_cols, axis=1)

# one-hot encode these features as factors
factor_cols = ['district']
train_data_race_blind_ohe, test_data_race_blind_ohe =util.get_dummies(train_data_race_blind,
                                                                      test_data_race_blind, factor_cols)


# ## Estimate the "race blind" multilayer perceptron model
# 

# In[26]:


# discard return vals; only print results
(_,_,_,_) = estimate_mlp(train_data_race_blind_ohe, train_labels, k_folds=5, max_iter=1000)


# > **KEY OBSERVATION**: the F1 score for the race-blind model declines further when we remove these features.  Of the models we have tested, the original "naive" model (with the most features) performs better than our race-blind model, or our model that excluded only zip and district.

# ## Experiment with dimensionality reduction via PCA
# Since manual feature selection performed poorly, resulting in a confidence interval of F1 spanning from 0 to 1 in both cases, it doesn't seem to be a promising approach.  Next, we experiment with Principal Component Analysis for dimensionality reduction, starting with the "naive" set of columns.

# In[27]:


# Determine the number of principal components to achieve 90% explained variance
n_pca = util.get_num_pcas(train_data_naive, var_explained=0.9)


# In[28]:


print('Using %d principal components' % (n_pca)) # currently n_pca=69

# discard return vals; only print results
(_,_,_,_) = estimate_mlp(train_data_naive_ohe, train_labels, n_pca=n_pca, k_folds=5, max_iter=1000)


# ## Use grid search to identify best set of hidden layer parameters
# Since the usage of PCA seemed to improve our F1 score (and tighten its confidence interval), we will proceed to try to optimize the hidden layer parameters while using PCA.

# In[29]:


# Running grid search for different combinations of neural network parameters is slow.
# If results already exist as a file, load them instead of re-running.
try:
    grid_search_results = pd.read_csv('cache_neuralnet/gridsearch_results.csv')
    print('Loaded grid search results from file.')
except FileNotFoundError:
    print('Performing grid search for best hidden layer parameters.')

    # We'll time it and report how long it took to run:
    start_time = time.time()

    # numbers of hidden nodes = these multipliers * # features
    n_features = train_data_naive_ohe.shape[1]
#     fraction = [0.25, 0.5]
    fraction = [0.25, 0.5, 1.0, 1.5, 2.0]
    n_layer_features = (int(f * n_features) for f in fraction)
    n_nodes = list(n_layer_features)

    # create list of tuples of hidden layer param permutations
    # only explore up to 4 hidden layers
    hl_param_candidates = []
    for h1 in n_nodes:
        hl_param_candidates.append((h1))
        for h2 in n_nodes:
            hl_param_candidates.append((h1,h2))
            for h3 in n_nodes:
                hl_param_candidates.append((h1,h2,h3))
                for h4 in n_nodes:
                    hl_param_candidates.append((h1,h2,h3,h4))
    
    # train an MLP model and perform cross-validation for each parameter set
    print('Estimating %d MLP models. This will take time!\n' % (len(hl_param_candidates)))
    tmp_results = []        
    for hl in hl_param_candidates:
        tmp_acc, tmp_acc_std, tmp_f1, tmp_f1_std = estimate_mlp(train_data_naive_ohe, train_labels, 
                                                                hidden_layers=hl, n_pca=n_pca,
                                                                k_folds=5, max_iter=1000, print_results=False)
        tmp_results.append((hl, tmp_acc, tmp_acc - 1.96 * tmp_acc_std, tmp_acc + 1.96 * tmp_acc_std,
                                    tmp_f1, tmp_f1 - 1.96 * tmp_f1_std, tmp_f1 + 1.96 * tmp_f1_std))

    # calculated elapsed time
    end_time = time.time()
    took = int(end_time - start_time)
    print("Grid search took {0:d} minutes, {1:d} seconds.".format(took // 60, took % 60))

    # convert results to a dataframe for easier display
    grid_search_results = pd.DataFrame(tmp_results)
    grid_search_results.columns=(['Hidden Layers','Accuracy','Acc Lower CI', 'Acc Upper CI','F1','F1 Lower CI','F1 Upper CI'])
    grid_search_results.to_csv('cache_neuralnet/gridsearch_results.csv', index=False)

# Display grid search results
grid_search_results.sort_values(by='F1', ascending=False)


# In[32]:


# put best grid search params into a varaiable
best_param_idx = grid_search_results['F1'].idxmax()

try:  # this is needed when loading from file
    best_hl_params = eval(grid_search_results['Hidden Layers'][best_param_idx])
except TypeError:  # eval isn't needed when results are still in memory
    best_hl_params = grid_search_results['Hidden Layers'][best_param_idx]
    
best_hl_params


# ## Calculate "out-of-sample" test set accuracy
# At this point we can use our "best" model parameters to classify our test set, and compare to true labels.
# 
# > NOTE: This code was left commented out until after hyperparameter optimization was complete.  

# In[63]:


# set up pipeline with optimal parameters
pipeline = make_pipeline(StandardScaler(with_mean=False), 
                     PCA(n_components=n_pca, random_state=207),
                     MLPClassifier(hidden_layer_sizes=best_hl_params,
                                   max_iter=1000, random_state=207))
pipeline.fit(train_data_naive_ohe, train_labels)
test_predict = pipeline.predict(test_data_naive_ohe)

print('Test set accuracy: %.2f\n' % (np.mean(test_predict==test_labels)))
print('Confusion matrix:')
cm = confusion_matrix(test_labels, test_predict)
print(cm)
tn, fp, fn, tp = cm.ravel()
print('True negatives: %d' % (tn))
print('True positives: %d' % (tp))
print('False negatives: %d' % (fn))
print('False positives: %d\n' % (fp))
print(classification_report(test_labels, test_predict))


# ## Analyze false positives to make recommendations to PASSNYC
# False positives are the schools that our model predicted to have a high SHSAT registration rate, but in reality they did not.  This suggests that they have a lot in common with the high registration schools, but for some reason fall short.  As a result, we believe these are good candidates for the PASSNYC organization to engage with, as investing in programs with these schools may be more highly to payoff with increase registration rates.  We will prioritize the schools based on features that align with the PASSNYC diversity-oriented mission.

# In[33]:


# recombine train and test data into an aggregate dataset
X_orig = pd.concat([train_data, test_data], sort=True)  # including all columns (need for display purposes)
X_best = pd.concat([train_data_naive_ohe, test_data_naive_ohe], sort=True)  # only columns from best model
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


# In[34]:


# Iterate through the Repeated Stratified K Fold, and and fill out the DataFrames
counter = 1
print('Please be patient...')
for train, test in rskf.split(X_best_npa, y_npa):
    # TODO: it might be possible to refact this into util if caller passes in a configured pipeline
    pipeline = make_pipeline(StandardScaler(with_mean=False), 
                         PCA(n_components=n_pca, random_state=207),
                         MLPClassifier(hidden_layer_sizes=best_hl_params,
                                       max_iter=1000, random_state=207))
    pipeline.fit(X_best_npa[train], y_npa[train])
    predicted_labels = pipeline.predict(X_best_npa[test])
    predictions.iloc[test, counter-1] = predicted_labels
    counter += 1
print('Completed')


# In[35]:


predictions.head()


# In[36]:


# Create columns for predictions and labels
predictions['1s'] = predictions.iloc[:,:50].sum(axis=1)
predictions['0s'] = (predictions.iloc[:,:50]==0).sum(axis=1)
predictions['true'] = y

# Create a table of all features along with the number of votes each received and the true value
X_predicted = pd.concat([X_orig, predictions['1s'], predictions['0s'],
                         pd.DataFrame(y)], axis=1, join_axes=[X_best.index])

# Rename the y columns to be more descriptive
X_predicted.rename(columns={0:"high_registrations"}, inplace=True)

# Sort by number of votes, most votes for 1 at the top and most votes for 0 at the bottom
X_predicted = X_predicted.sort_values(by=['1s', '0s'], ascending=[False, True])


# In[37]:


# Retain only the columns of interest for PASSNYC prioritization
final_features = ['1s',
                  'dbn',
                  'school_name',
                  'economic_need_index',
                  'grade_7_enrollment',
                  'num_shsat_test_takers',
                  'pct_test_takers',
                  'percent_black__hispanic'
              ]
df_passnyc = X_predicted[final_features]

# Determine the number of test takers this school would have needed to meet
#   the median percentage of high_registrations
median_pct = np.median(X_orig[y==1]['pct_test_takers'])/100
target_test_takers = np.multiply(df_passnyc['grade_7_enrollment'], median_pct)

# Subtract the number of actual test takers from the hypothetical minimum number
delta = target_test_takers - df_passnyc['num_shsat_test_takers']

# Multiply the delta by the minority percentage of the school to determine how many minority
#   students did not take the test
minority_delta = np.round(np.multiply(delta, df_passnyc['percent_black__hispanic']/100),0).astype(int)
df_passnyc = df_passnyc.assign(minority_delta=minority_delta)

# Multiply the minority delta by the percentage of 1 votes to get a confidence-adjusted 'score'
df_passnyc = df_passnyc.assign(score=np.multiply(df_passnyc['minority_delta'], df_passnyc['1s']/10))

# sort descending, and filter to schools with more than five minority students
df_passnyc = df_passnyc.sort_values(by='score', ascending=False)

# Create a rank order column
df_passnyc.insert(0, 'rank', range(1,df_passnyc.shape[0]+1))

# Write to CSV
df_passnyc.to_csv('results/results.neuralnet.csv')

df_passnyc


# ## Post-hoc comparison of prioritization score vs. economic need index
# Even though economic need index was not an explicit factor in our post-classification prioritization scoring/ranking system, it is interesting to observe that there is some correlation:

# In[50]:


x = df_passnyc['score']
y = df_passnyc['economic_need_index']
sns.regplot(x='score', y='economic_need_index', data=df_passnyc,
           fit_reg=True, x_jitter=1, scatter_kws={'alpha': 0.5, 's':4})
# plt.scatter(x, y)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel('PASSNYC Priorization "Score"')
plt.ylabel('Economic Need Index')
plt.grid()
plt.show()

