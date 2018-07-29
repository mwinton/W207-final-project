
# coding: utf-8

# # Exploratory Data Analysis
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton
# 
# #### W207-4-Summer 2018 Final Project
# 
# [Return to project overview](final_project_overview.ipynb)

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import util

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# Get train-test split
train_data, test_data, train_labels, test_labels = util.read_data()

# Concatenate training features and labels into one dataframe
Xy_train = pd.concat([train_data, train_labels], axis=1)


# ## Correlation Matrix: key non-demographic features
# 
# As we expect many of our features to be highly correlated, looking at a visual representation of the correlation matrix is a useful step in our EDA.  In this first plot, we intentionally omit features either directly representing, or closely related to demographic features.  

# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')

# choose key features for correlation matrix
corr_features = ['grade_7_enrollment',
                 'community_school', 
                 'student_attendance_rate',
                 'percent_of_students_chronically_absent',
                 'rigorous_instruction_percent', 
                 'collaborative_teachers_percent', 
                 'supportive_environment_percent',
                 'effective_school_leadership_percent',
                 'strong_family_community_ties_percent',
                 'trust_percent',
                 'student_achievement_rating',
                 'average_ela_proficiency',
                 'average_math_proficiency',
                 'grade_7_ela_all_students_tested',
                 'grade_7_ela_4s_all_students',
                 'grade_7_math_all_students_tested',
                 'grade_7_math_4s_all_students',
                 'average_class_size_english', 
                 'average_class_size_math',
                 'average_class_size_science',
                 'average_class_size_social_studies',
                 'school_pupil_teacher_ratio',
                 'high_registrations']


def draw_heatmap(df, fig_name):
    corr_matrix = df.corr(method='pearson')
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    # plot heatmap and also save to disk
    plt.rcParams['figure.figsize'] = [12, 8]
    sns.heatmap(corr_matrix, annot=False, fmt="g",
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns,
                center=0, linewidths=0.5, cmap="coolwarm", robust=True, mask=mask)
    plt.savefig(fig_name, bbox_inches='tight')

draw_heatmap(Xy_train[corr_features], 'plots/corr_matrix_key_features.png')


# From this correlation matrix, we see that in general, feature related to test scores or academic proficiency are positively correlated with a high SHSAT registration rate.  Two features in our dataset are most negatively correlated with registration rate: the percentage of students which are chronically absent, as well as the "community school" indicator.  The former is very much intuitive, but the latter fact may be an noteworthy learning from this analysis.  It's also somewhat interesting to see that a _higher_ student-to-teacher ratio correlates positively with registration rate.

# ## Correlation Matrix: demographic features
# 
# As the lack of diversity in the SHSAT registrations is part of our original problem statement, it is also interesting to look at the existing correlation between demographic-related features and test registrations.

# In[25]:


# choose key features for correlation matrix
demog_features =  ['economic_need_index',
                   'percent_ell',
                   'percent_asian',
                   'percent_black', 
                   'percent_hispanic', 
                   'percent_black__hispanic',
                   'percent_white', 
                   'grade_7_ela_4s_black_or_african_american',
                   'grade_7_ela_4s_hispanic_or_latino',
                   'grade_7_ela_4s_multiracial',
                   'grade_7_ela_4s_limited_english_proficient',
                   'grade_7_ela_4s_economically_disadvantaged',
                   'grade_7_math_4s_black_or_african_american',
                   'grade_7_math_4s_hispanic_or_latino',
                   'grade_7_math_4s_multiracial',
                   'grade_7_math_4s_limited_english_proficient',
                   'grade_7_math_4s_economically_disadvantaged',
                   'high_registrations']

draw_heatmap(Xy_train[demog_features], 'plots/corr_matrix_demographics.png')


# We see that percent Asian and percent white correlate positively with high SHSAT registration rates, whereas percent black and Hispanic correlate negatively with the registration rate.  This is in agreement with the original problem statement that PASSNYC posed.  Additionally, schools with a higher economic need index correlate with lower registration rates.  However, even in light of this fact, it is particularly interesting to note that schools with a high proportion of economically disadvantage students scoring 4's on their ELA and Math exams tend to have higher registration rates, as well.
