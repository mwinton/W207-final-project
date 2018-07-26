
# coding: utf-8

# # K Nearest Neighbors
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# In this notebook, we attempt to classify the PASSNYC data via K-Nearest Neighbors algorithm.
# 
# ### Reading data
# Let us do some initial imports and set up the data.

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from functools import partial
from sklearn.model_selection import train_test_split

# Get train-test split
train_data, test_data, train_labels, test_labels = util.read_data()

# From the training data, create a validate data split as well
train_data, validate_data, train_labels, validate_labels =     util.train_test_split(train_data, train_labels)

print("Train data shape: %s" % str(train_data.shape))
print("Validate data shape: %s" % str(validate_data.shape))
print("Test data shape: %s" % str(test_data.shape))
train_data.head()


# ### Feature selection
# 
# We will now select some features from the above dataset.
# 
# Let us shortlist some interesting columns:
# * dbn
# * rigorous_instruction_percent
# * rigorous_instruction_rating
# * collaborative_teachers_percent
# * collaborative_teachers_rating
# * supportive_environment_percent
# * supportive_environment_rating
# * effective_school_leadership_percent
# * effective_school_leadership_rating
# * strong_family_community_ties_percent
# * strong_family_community_ties_rating
# * trust_percent
# * trust_rating
# * student_achievement_rating
# * average_ela_proficiency
# * average_math_proficiency
# * grade_7_ela_all_students_tested
# * grade_7_ela_4s_all_students
# * grade_7_math_all_students_tested
# * grade_7_math_4s_all_students
# * average_class_size_english
# * average_class_size_math
# * school_pupil_teacher_ratio
# * student_attendance_rate
# 
# We ignore the following demographic indicators:
# * school_name
# * zip
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

perf_train_data = train_data[["rigorous_instruction_percent","rigorous_instruction_rating","collaborative_teachers_percent","collaborative_teachers_rating","supportive_environment_percent","supportive_environment_rating","effective_school_leadership_percent","effective_school_leadership_rating","strong_family_community_ties_percent","strong_family_community_ties_rating","trust_percent","trust_rating","student_achievement_rating","average_ela_proficiency","average_math_proficiency","grade_7_ela_all_students_tested","grade_7_ela_4s_all_students","grade_7_math_all_students_tested","grade_7_math_4s_all_students","average_class_size_english","average_class_size_math","school_pupil_teacher_ratio","student_attendance_rate"]]


# ### PCA
# 
# We will first run PCA to see if we can reduce the dimensions significantly.  It seems like 3 dimensions are enough for variance ratio to be > 0.7.

# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())
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


# ### Univariate Model
# 
# Next, we try to select N-best features, based on univariate statistical tests.  We use the $\chi^2$ test.

# In[5]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

pipeline = make_pipeline(MinMaxScaler(), 
                         SelectKBest(chi2, k=5))
pipeline.fit_transform(perf_train_data_nonull, train_labels)
selected_features = pipeline.steps[1][1].get_support()
perf_train_data_nonull.columns[selected_features]


# The univariate _KBest_ model selects the following features, for $K = 5$:
# * Average ELA proficiency
# * Average math proficiency
# * Grade 7 ELA all students
# * Grade 7 math 4S all students
# * Student attendance rate

# ### Linear Model
# 
# We next run a linear model with L1 penalty and regularization (C) to select features.  Let us see what features it selects.

# In[8]:


from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

pipeline = make_pipeline(StandardScaler(), 
                         SelectFromModel(LinearSVC(C=0.05, penalty='l1', dual=False, random_state=207)))
pipeline.fit_transform(perf_train_data_nonull, train_labels)
selected_features = pipeline.steps[1][1].get_support()
perf_train_data_nonull.columns[selected_features]


# Therefore, the SVM model selects the following features:
# 
# * Collaborative teachers rating
# * Average math proficiency
# * Grade 7 ELA 4S all students
# * Grade 7 math 4S all students
# * Student attendance rate
# 

# ### Tree-based Model
# 
# Let us now run a tree-based estimator to see which features it selects.

# In[9]:


from sklearn.ensemble import ExtraTreesClassifier

pipeline = make_pipeline(StandardScaler(), 
                         SelectFromModel(ExtraTreesClassifier()))
pipeline.fit_transform(perf_train_data_nonull, train_labels)
selected_features = pipeline.steps[1][1].get_support()
perf_train_data_nonull.columns[selected_features]


# The model selects the following features:
# * Average ELA proficiency
# * Average math proficiency
# * Grade 7 ELA 4S all students
# * Grade 7 math all students
# * Grade 7 math 4S all students
# * School pupil teacher ratio

# ### Final selection
# 
# Considering all the features above, we can now select a final set of features.
# 
# The following are in 1+ models, so we will select them:
# * Grade 7 math 4S all students
# * Grade 7 ELA 4S all students
# * Average math proficiency
# * Average ELA proficiency
# * Student attendance rate
# 
# The following are columns we will keep as a backup:
# * School pupil teacher ratio
# * Grade 7 math all students
# * Collaborative teachers rating
# 

# ### K-Nearest Neighbors Classification
# 
# We will now run KNN prediction on the dataset.

# In[11]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

selected_features = ['grade_7_math_4s_all_students',
                     'grade_7_ela_4s_all_students', 
                     'average_math_proficiency',
                     'average_ela_proficiency',
                     'student_attendance_rate']
perf_train_data_nonull_knn = perf_train_data_nonull[selected_features]

scaler = StandardScaler().fit(perf_train_data_nonull_knn)
rescaledX = scaler.transform(perf_train_data_nonull_knn)
clf = KNeighborsClassifier()
clf.fit(rescaledX, train_labels)

perf_validate_data_nonull_knn = validate_data.fillna(validate_data.mean())[selected_features]
scaler = StandardScaler().fit(perf_validate_data_nonull_knn)
rescaledXval = scaler.transform(perf_validate_data_nonull_knn)
y = clf.predict(rescaledXval)

print(classification_report(y, validate_labels, target_names=['Predict low-registrations', 'Predict high-registrations']))


# ### Conclusion
# 
# The KNN model gives good results for low-registrations, but not for high-registrations.  There is also sparse data for "high registrations", as shown by the "support" column.  We may be better off using an ensemble technique to avoid giving undue weightage to a single classification algorithm.
