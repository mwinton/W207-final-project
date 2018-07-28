
# coding: utf-8

# # Logistic Regression Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# In[ ]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from util import our_train_test_split

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().magic('matplotlib inline')


# In[ ]:


# Import cleaned dataset
merged_df = pd.read_csv('data_merged/combined_data_2018-07-18.csv')

merged_df['number_enrolled'] = np.array((merged_df['num_shsat_test_takers']*100)/merged_df['pct_test_takers']).astype(int)

# Keep the numeric columns.
features_to_keep = [
                    'high_registrations',
                    #'district', 
                    #'zip',
                    'community_school', 
                    'economic_need_index', 
                    #'school_income_estimate',
                    'percent_ell', 
                    'percent_asian', 
                    'percent_black', 
                    'percent_hispanic',
                    'percent_black__hispanic', 
                    'percent_white', 
                    'student_attendance_rate',
                    'percent_of_students_chronically_absent',
                    'rigorous_instruction_percent', 
                    'rigorous_instruction_rating',
                    'collaborative_teachers_percent', 
                    'collaborative_teachers_rating',
                    'supportive_environment_percent', 
                    'supportive_environment_rating',
                    'effective_school_leadership_percent',
                    'effective_school_leadership_rating',
                    'strong_family_community_ties_percent',
                    'strong_family_community_ties_rating', 
                    'trust_percent', 
                    'trust_rating',
                    'student_achievement_rating', 
                    'average_ela_proficiency',
                    'average_math_proficiency', 
                    'grade_7_ela_all_students_tested',
                    'grade_7_ela_4s_all_students',
                    'grade_7_ela_4s_american_indian_or_alaska_native',
                    'grade_7_ela_4s_black_or_african_american',
                    'grade_7_ela_4s_hispanic_or_latino',
                    'grade_7_ela_4s_asian_or_pacific_islander', 
                    'grade_7_ela_4s_white',
                    'grade_7_ela_4s_multiracial',
                    'grade_7_ela_4s_limited_english_proficient',
                    'grade_7_ela_4s_economically_disadvantaged',
                    'grade_7_math_all_students_tested', 
                    'grade_7_math_4s_all_students',
                    'grade_7_math_4s_american_indian_or_alaska_native',
                    'grade_7_math_4s_black_or_african_american',
                    'grade_7_math_4s_hispanic_or_latino',
                    'grade_7_math_4s_asian_or_pacific_islander', 
                    'grade_7_math_4s_white',
                    'grade_7_math_4s_multiracial',
                    'grade_7_math_4s_limited_english_proficient',
                    'grade_7_math_4s_economically_disadvantaged',
                    'number_of_students_english', 
                    'number_of_students_math',
                    'number_of_students_science', 
                    'number_of_students_social_studies',
                    'number_of_classes_english', 
                    'number_of_classes_math',
                    'number_of_classes_science', 
                    'number_of_classes_social_studies',
                    'average_class_size_english', 
                    'average_class_size_math',
                    'average_class_size_science',
                    'average_class_size_social_studies',
                    'school_pupil_teacher_ratio',
                    'number_enrolled'
                   ]

X = merged_df[features_to_keep]
X.head()


# Split DataFrame into data and labels

# In[ ]:


y = X['high_registrations']
X = X.drop(['high_registrations'], axis=1)


# ## Deal with NaNs
# Some columns have NaNs.  We'll try imputing these to the mean.

# In[ ]:


from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=np.nan, strategy='mean')
X_i = pd.DataFrame(imp.fit_transform(X))
X_i.columns = X.columns
X_i.index = X.index
X_i.head()


# In[ ]:


X_pos = X_i[y==1]
X_neg = X_i[y==0]


# ## Split the dataset into train and test splits

# In[ ]:


from functools import partial
from sklearn.model_selection import train_test_split
import util

train_data, test_data, train_labels, test_labels = util.our_train_test_split(X_i, y, stratify = y)
train_data.head()


# ## Hyperparameter Tuning
# Find the optimal C-score

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)

lr = LogisticRegression(random_state=207)
penalty = ['l1', 'l2']
C = [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 1.000, 2.000, 3.000, 4.000]
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0)
best_model = clf.fit(train_data_scaled, train_labels)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])



# In[ ]:


from sklearn import metrics
from sklearn.metrics import classification_report
c_values = {'C': [0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 1.000, 2.000, 3.000, 4.000]}
best_c = 0
top_f1 = 0
for c in c_values['C']:
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    log = LogisticRegression(C=c, penalty='l2', random_state=207)
    log.fit(train_data_scaled, train_labels.values.ravel())
    log_predicted_labels = log.predict(test_data_scaled)
    log_f1 = metrics.f1_score(test_labels, log_predicted_labels, average='weighted')
    log_accuracy = metrics.accuracy_score(test_labels, log_predicted_labels)
    if log_f1 > top_f1:
        top_f1 = log_f1
        best_c = c
    print("F1 score for C={}: {:.4f}    Accuracy: {:.4f}".format(c, log_f1, log_accuracy))

print("\nThe best C value is {} with an F1 of {:.4f}".format(best_c, top_f1))


# ## Make Pipeline and K-fold validation

# In[ ]:


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression(C=.1, penalty='l1', random_state=207))
k_folds = 10
cv_scores = cross_validate(pipe, train_data, train_labels, cv=k_folds, scoring=['accuracy','f1'])

# display accuracy with 95% confidence interval
cv_accuracy = cv_scores['test_accuracy']
print ('With %d-fold cross-validation, accuracy is: %.3f (95%% CI from %.3f to %.3f).' %
       (k_folds, cv_accuracy.mean(), cv_accuracy.mean() - 1.96 * cv_accuracy.std(),
        cv_accuracy.mean() + 1.96 * cv_accuracy.std()))

# display F1 score with 95% confidence interval
cv_f1 = cv_scores['test_f1']
print ('The F1 score is: %.3f (95%% CI from %.3f to %.3f).' %
       (cv_f1.mean(), cv_f1.mean() - 1.96 * cv_f1.std(),
        cv_f1.mean() + 1.96 * cv_f1.std()))


# ## Examine coefficients

# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold

# Run coefficient analysis on 100% of the data
np_train_data = np.array(scaler.fit_transform(X_i))
np_train_labels = np.array(y)

# Run k-fold cross-validation with 5 folds 10 times, which means every school is predicted 10 times.
folds = 5
repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats= repeats, random_state=207)
fold_list = []

# Build two dataframes from the results, with columns for each k-fold
for f in range(1, (folds*repeats)+1):
    fold_list.append('k{}'.format(f))
coefs = pd.DataFrame(index=train_data.columns, columns=fold_list)
predictions = pd.DataFrame(index=merged_df.index, columns = fold_list)

# Iterate through the Repeated Stratified K Fold, and and fill out the DataFrames
counter = 1
for train, test in rskf.split(np_train_data, np_train_labels):
    log = LogisticRegression(C=1, penalty='l2', random_state=207)
    log.fit(np_train_data[train], np_train_labels[train])
    predicted_labels = log.predict(np_train_data[test])
    coefs['k{}'.format(counter)] = log.coef_[0]
    predictions.iloc[test, counter-1] = predicted_labels
    counter += 1

# Find the average coefficient across all 50 regressions, and sort descending
coefs['avg'] = coefs.mean(axis=1)
sorted_coefs = coefs.sort_values(by='avg', ascending=False)
sorted_coefs


# ## Distributions
# ### Most positively-influential features

# In[ ]:


# Get the top and bottom 5 most influential coefficients
top_features, bottom_features = list(sorted_coefs.index[:6]), list(sorted_coefs.index[-6:])
fig = plt.figure(figsize=(20,60))
for c in top_features:
    ax = fig.add_subplot(6,2,top_features.index(c)+1)
    ax.set_title('Distribution for {}'.format(c), fontsize=17)
    ax.hist(X_pos[c], bins=20, alpha=0.5, label="high registrants", density=True)
    ax.hist(X_neg[c], bins=20, alpha=0.5, label="low registrants", density=True)
    ax.legend()
plt.show()


# ### Most negatively-influential features

# In[ ]:


fig = plt.figure(figsize=(20,60))
for c in bottom_features:
    ax = fig.add_subplot(6,2,bottom_features.index(c)+1)
    ax.set_title('Distribution for {}'.format(c), fontsize=17)
    ax.hist(X_pos[c], bins=20, alpha=0.5, label="high registrants", density=True)
    ax.hist(X_neg[c], bins=20, alpha=0.5, label="low registrants", density=True)
    ax.legend()
plt.show()


# ## Examining Wrong Answers
# We can build a table of all the kfold predictions, and see the degree to which the model got each school right or wrong

# In[ ]:


import seaborn as sns
sns.regplot(X_pos.grade_7_math_4s_hispanic_or_latino, X_pos.grade_7_ela_4s_hispanic_or_latino)
sns.regplot(X_neg.grade_7_math_4s_hispanic_or_latino, X_neg.grade_7_ela_4s_hispanic_or_latino)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax = sns.regplot(X_i.grade_7_math_4s_hispanic_or_latino, y, logistic=True, label="Math")
ax = sns.regplot(X_i.grade_7_ela_4s_hispanic_or_latino, y, logistic=True, label="English")
ax.legend()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax = sns.regplot(X_i.number_enrolled, y, logistic=True, label="Number Enrolled")
ax.legend()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(X_pos.grade_7_math_4s_hispanic_or_latino, X_pos.grade_7_ela_4s_hispanic_or_latino, color='blue', label="high registrants")
ax.scatter(X_neg.grade_7_math_4s_hispanic_or_latino, X_neg.grade_7_ela_4s_hispanic_or_latino, color='red', label="low registrants")
ax.set_xlabel('Grade 7 Math 4s, Hispanic or Latino', fontsize=15)
ax.set_ylabel('Grade 7 ELA 4s, Hispanic or Latino', fontsize=15)
plt.show


# In[ ]:


predictions['1s'] = predictions.iloc[:,:50].sum(axis=1)
predictions['0s'] = (predictions.iloc[:,:50]==0).sum(axis=1)
predictions['true'] = y


# In[ ]:


# Create a table of raw results, the vote for truth
X_predicted = pd.concat([X_i, predictions['1s'], predictions['0s'], y], axis=1, join_axes=[X_i.index])
X_predicted = X_predicted.sort_values(by=['1s', '0s'], ascending=[False, True])


scaled_X_predicted = scaler.fit_transform(X_predicted)
avg_coefs = np.array(coefs['avg'])
weighted_values = np.multiply(scaled_X_predicted, avg_coefs)
X_result_weighted = X_predicted.copy()
X_result_weighted.iloc[:, :-3] = weighted_values
X_result_weighted = X_result_weighted.iloc[:, :]
drop_cols = []
for c in X_result_weighted.columns:
    if X_result_weighted[c].max() - X_result_weighted[c].min() < .8:
        drop_cols.append(c)
X_result_weighted_trimmed = X_result_weighted.drop(drop_cols, axis=1)
X_result_weighted_trimmed.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(18,200))
im = ax.imshow(X_result_weighted_trimmed, cmap='viridis')
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(X_result_weighted_trimmed.columns)))
ax.set_yticks(np.arange(len(X_result_weighted_trimmed.index)))
ax.set_xticklabels(X_result_weighted_trimmed.columns)
ax.set_yticklabels(X_result_weighted_trimmed.index)
plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")

#for i in range(len(X_result_weighted_trimmed.index)):
#    for j in range(len(X_result_weighted_trimmed.columns[1:])):
#        text = ax.text(j, i, round(X_predicted.loc[X_result_weighted_trimmed.index[i], X_result_weighted_trimmed.columns[j+1]], 1),
#                       ha="center", va="center", color="w")
plt.show()


# It is perhaps most useful to examine the false positives - that is, schools that did NOT have high SHSAT registrations, but that the model thought SHOULD have.  We'll put the threshhold at 5 or more incorrect "true" classifications, and rank them in descending order (ie, the schools the model got most consistently wrong at the top).

# In[ ]:


false_positives = predictions[predictions['true']==0]
false_positives = false_positives[false_positives['1s'] > 5].sort_values(by='1s', ascending=False)['1s']
false_positives
fp_result = pd.concat([false_positives, merged_df.iloc[false_positives.index]], axis=1, join_axes=[false_positives.index])
fp_result


# In[ ]:


# Just the columns of interest
fp_features = ['1s',
              'dbn',
              'school_name',
              'number_enrolled',
              'num_shsat_test_takers']
# Convert the percent columns to proper floats
pct_features = ['pct_test_takers',
               'percent_black__hispanic']
df_pct = np.multiply(fp_result[pct_features], .01)
# Merge these seven columns to one DataFram
df_false_positives = pd.concat([fp_result[fp_features], df_pct], axis=1)

# Determine the number of test takers this school would have needed to meet the median percentage of high_registrations
median_pct = np.median(merged_df[merged_df['high_registrations']==1]['pct_test_takers'])/100
predicted_test_takers = np.multiply(df_false_positives['number_enrolled'], median_pct)

# Subtract the number of actual test takers from the hypothetical minimum number
delta = predicted_test_takers - df_false_positives['num_shsat_test_takers']

# Multiply the delta by the minority percentage of the school to determine how many minority students did not take the test
minority_delta = np.round(np.multiply(delta, df_false_positives['percent_black__hispanic']), 0)

# Add this number to the dataframe, sort descending, and filter to schools with more than five minority students
df_false_positives['minority_delta'] = minority_delta
df_false_positives = df_false_positives[df_false_positives['minority_delta'] > 5]                     .sort_values(by='minority_delta', ascending=False)
# Create a rank order column
df_false_positives.insert(0, 'rank', range(1,df_false_positives.shape[0]+1))
# Write to CSV
df_false_positives.to_csv('results/results.logreg.csv')
df_false_positives


# Determine p-value for each value in the false positives group

# In[ ]:


scaled_fp_X = scaler.fit_transform(fp_result.iloc[:,1:])
avg_coefs = np.array(coefs['avg'])
weighted_values = np.multiply(scaled_fp_X, avg_coefs)
fp_result_weighted = fp_result.copy()
fp_result_weighted.iloc[:, 1:] = weighted_values
# Drop the columns that don't have any evidence of influential values
drop_cols = []
for c in fp_result_weighted.columns:
    if fp_result_weighted[c].max() - fp_result_weighted[c].min() < .8:
        drop_cols.append(c)
fp_result_weighted_trimmed = fp_result_weighted.drop(drop_cols, axis=1)
fp_result_weighted_trimmed


# In[ ]:


fig, ax = plt.subplots(figsize=(18,18))
im = ax.imshow(fp_result_weighted_trimmed.iloc[:,1:], cmap='viridis')
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(fp_result_weighted_trimmed.columns[1:])))
ax.set_yticks(np.arange(len(fp_result_weighted_trimmed.index)))
ax.set_xticklabels(fp_result_weighted_trimmed.columns[1:])
ax.set_yticklabels(fp_result_weighted_trimmed.index)
plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")

for i in range(len(fp_result_weighted_trimmed.index)):
    for j in range(len(fp_result_weighted_trimmed.columns[1:])):
        text = ax.text(j, i, round(fp_result.loc[fp_result_weighted_trimmed.index[i], fp_result_weighted_trimmed.columns[j+1]], 1),
                       ha="center", va="center", color="w")
plt.show()


# In[ ]:


false_negatives = predictions[predictions['true']==1]
false_negatives = false_negatives[false_negatives['0s'] > 5].sort_values(by='0s', ascending=False)['0s']
false_negatives
fn_result = pd.concat([false_negatives, merged_df.iloc[false_negatives.index]], axis=1, join_axes=[false_negatives.index])
fn_result


# In[ ]:


scaled_fn_X = scaler.fit_transform(fn_result.iloc[:,1:])
weighted_values = np.multiply(scaled_fn_X, avg_coefs)
fn_result_weighted = fn_result.copy()
fn_result_weighted.iloc[:, 1:] = weighted_values
# Drop the columns that don't have any evidence of influential values
drop_cols = []
for c in fn_result_weighted.columns:
    if fn_result_weighted[c].max() - fn_result_weighted[c].min() < .8:
        drop_cols.append(c)
fn_result_weighted_trimmed = fn_result_weighted.drop(drop_cols, axis=1)
fn_result_weighted_trimmed


# In[ ]:


fig, ax = plt.subplots(figsize=(18,18))
im = ax.imshow(fn_result_weighted_trimmed.iloc[:,1:], cmap='viridis')
ax.xaxis.tick_top()
ax.set_xticks(np.arange(len(fn_result_weighted_trimmed.columns[1:])))
ax.set_yticks(np.arange(len(fn_result_weighted_trimmed.index)))
ax.set_xticklabels(fn_result_weighted_trimmed.columns[1:])
ax.set_yticklabels(fn_result_weighted_trimmed.index)
plt.setp(ax.get_xticklabels(), rotation=90, ha="left", va="center", rotation_mode="anchor")

for i in range(len(fn_result_weighted_trimmed.index)):
    for j in range(len(fn_result_weighted_trimmed.columns[1:])):
        text = ax.text(j, i, round(fn_result.loc[fn_result_weighted_trimmed.index[i], fn_result_weighted_trimmed.columns[j+1]], 1),
                       ha="center", va="center", color="w")
plt.show()

