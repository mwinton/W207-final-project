
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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import cleaned dataset
merged_df = pd.read_csv('data_merged/combined_data_2018-07-18.csv')

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
                    'school_pupil_teacher_ratio'
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


# ## Split the dataset into train and test splits

# In[ ]:


from functools import partial
from sklearn.model_selection import train_test_split
import util

train_data, test_data, train_labels, test_labels = our_train_test_split(X_i, y, stratify = y)
train_data.head()


# ## Hyperparameter Tuning
# Find the optimal C-score

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
c_values = {'C': [0.010, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 1.000, 2.000, 3.000, 4.000]}
for c in c_values['C']:
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    log = LogisticRegression(C=c, penalty='l2', random_state=207)
    log.fit(train_data_scaled, train_labels.values.ravel())
    log_predicted_labels = log.predict(test_data_scaled)
    log_f1 = metrics.f1_score(test_labels, log_predicted_labels, average='weighted')
    log_accuracy = metrics.accuracy_score(test_labels, log_predicted_labels)
    print("F1 score for C={}: {:.4f}    Accuracy: {:.4f}".format(c, log_f1, log_accuracy))


# ## Make Pipeline and K-fold validation

# In[ ]:


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1, penalty='l2', random_state=207))
k_folds = 10
cv_scores = cross_validate(pipe, train_data, train_labels, cv=k_folds, scoring=['accuracy','f1'])


# In[ ]:


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
scaler = StandardScaler()
np_train_data = np.array(scaler.fit_transform(X_i))
np_train_labels = np.array(y)
folds = 5
repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats= repeats, random_state=207)
fold_list = []
for f in range(1, (folds*repeats)+1):
    fold_list.append('k{}'.format(f))
coefs = pd.DataFrame(index=train_data.columns, 
                     columns=fold_list)
predictions = pd.DataFrame(index=merged_df.index, columns = fold_list)
counter = 1

for train, test in rskf.split(np_train_data, np_train_labels):
    log = LogisticRegression(C=1, penalty='l2', random_state=207)
    log.fit(np_train_data[train], np_train_labels[train])
    predicted_labels = log.predict(np_train_data[test])
    coefs['k{}'.format(counter)] = log.coef_[0]
    predictions.iloc[test, counter-1] = predicted_labels
    counter += 1

coefs['avg'] = coefs.mean(axis=1)
coefs.sort_values(by='avg', ascending=False)


# ## Examining Wrong Answers
# We can build a table of all the kfold predictions, and see the degree to which the model got each school right or wrong

# In[ ]:


predictions['1s'] = predictions.iloc[:,:50].sum(axis=1)
predictions['0s'] = (predictions.iloc[:,:50]==0).sum(axis=1)
predictions['true'] = y


# It is perhaps most useful to examine the false positives - that is, schools that did NOT have high SHSAT registrations, but that the model thought SHOULD have.  We'll put the threshhold at 5 or more incorrect "true" classifications, and rank them in descending order (ie, the schools the model got most consistently wrong at the top).

# In[ ]:


false_positives = predictions[predictions['true']==0]
false_positives = false_positives[false_positives['1s'] > 5].sort_values(by='1s', ascending=False)['1s']
false_positives
result = pd.concat([false_positives, merged_df.iloc[false_positives.index]], axis=1, join_axes=[false_positives.index])
result

