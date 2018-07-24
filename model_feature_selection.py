
# coding: utf-8

# # K-Nearest Neighbors Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)

# Let us do some initial imports and set up the data.

# In[ ]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from functools import partial
from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = util.read_data()
train_data


# Let us shortlist some interesting columns:
# * dbn
# * num_shsat_test_takers
# * offers_per_student
# * pct_test_takers
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
# Demographic indicators
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

# In[ ]:


# To generate this list again:
# Take above (markdown) list and store it in say ~/tmp/col_list.  Then:
# cat  ~/tmp/col_list | cut -d" " -f2 | sed -E 's/^(.*)$/"\1"/' | tr '\n' ', '

perf_train_data = train_data[["num_shsat_test_takers","offers_per_student","pct_test_takers","rigorous_instruction_percent","rigorous_instruction_rating","collaborative_teachers_percent","collaborative_teachers_rating","supportive_environment_percent","supportive_environment_rating","effective_school_leadership_percent","effective_school_leadership_rating","strong_family_community_ties_percent","strong_family_community_ties_rating","trust_percent","trust_rating","student_achievement_rating","average_ela_proficiency","average_math_proficiency","grade_7_ela_all_students_tested","grade_7_ela_4s_all_students","grade_7_math_all_students_tested","grade_7_math_4s_all_students","average_class_size_english","average_class_size_math","school_pupil_teacher_ratio","student_attendance_rate"]]


# In[ ]:


from sklearn.decomposition import PCA
perf_train_data.info()
perf_train_data_nonull = perf_train_data.fillna(perf_train_data.mean())
cum_explained_variance_ratios = []
#perf_train_data_nonull.info()
for n in range(1, 15):
    pca = PCA(n_components=n, random_state=207)
    pca.fit(perf_train_data_nonull)
    print(pca.explained_variance_ratio_)
    cum_explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

import seaborn as sns
sns.set()
plt.plot(np.array(cum_explained_variance_ratios))


# In[ ]:


pca = PCA(n_components=2, random_state=207)
pca.fit(perf_train_data_nonull)
perf_train_data_nonull_pca = pca.transform(perf_train_data_nonull)
plt.scatter(perf_train_data_nonull_pca[:, 0], perf_train_data_nonull_pca[:, 1])
plt.axis('equal')
plt.show()

pca = PCA(n_components=1, random_state=207)
pca.fit(perf_train_data_nonull)
perf_train_data_nonull_pca = pca.transform(perf_train_data_nonull)
plt.plot(perf_train_data_nonull_pca)
plt.axis('equal')
plt.show()


# In[ ]:




