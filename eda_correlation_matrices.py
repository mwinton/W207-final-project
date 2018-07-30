
# coding: utf-8

# # Exploratory Data Analysis
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton
# 
# #### W207-4-Summer 2018 Final Project
# 
# [Return to project overview](final_project_overview.ipynb)

# In[14]:


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
# convert train_labels into a dataframe in order to concatenate it
train_labels_df = pd.DataFrame(train_labels)
train_labels_df.columns=['high_registrations']

# Concatenate training features and labels into one dataframe
Xy_train = pd.concat([train_data, train_labels_df], axis=1)


# ## Verifying correlation between "equivalent" percent and rating columns
# The definitions of these "percent" and "rating" columns on Kaggle are identical, leading us to question whether these were duplicate columns, so we plotted them against each other. 

# In[15]:


plt.rcParams['figure.figsize'] = [12, 3]
plt.scatter(train_data['rigorous_instruction_percent'], train_data['rigorous_instruction_rating'])
plt.title('rigorous_instruction')
plt.show()
plt.scatter(train_data['collaborative_teachers_percent'], train_data['collaborative_teachers_rating'])
plt.title('collaborative_teachers')
plt.show()
plt.scatter(train_data['supportive_environment_percent'], train_data['supportive_environment_rating'])
plt.title('supportive_environment')
plt.show()
plt.scatter(train_data['effective_school_leadership_percent'], train_data['effective_school_leadership_rating'])
plt.title('effective_school_leadership')
plt.show()
plt.scatter(train_data['strong_family_community_ties_percent'], train_data['strong_family_community_ties_rating'])
plt.title('strong_family_community_ties')
plt.show()
plt.scatter(train_data['trust_percent'], train_data['trust_rating'])
plt.title('trust')
plt.show()


# It turns out we cannot consider the `percent` and `rating` columns to be duplicates because, for example an 85% on most plots could be a 2, 3, or 4 rating.  They are correlated, but are not perfectly multicollinear; as a result, we will keep both sets of features in the dataset.

# ## Correlation Matrix: key non-demographic features
# 
# As we expect many of our features to be highly correlated, looking at a visual representation of the correlation matrix is a useful step in our EDA.  In this first plot, we intentionally omit features either directly representing, or closely related to demographic features.  

# In[17]:


get_ipython().magic('matplotlib inline')

# choose key features for correlation matrix
corr_features = ['grade_7_enrollment',
                 'community_school', 
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
                center=0, linewidths=0.5, cmap="coolwarm", mask=mask)
    plt.savefig(fig_name, bbox_inches='tight')

draw_heatmap(Xy_train[corr_features], 'plots/corr_matrix_key_features.png')


# From this correlation matrix, we see that in general, feature related to test scores or academic proficiency are positively correlated with a high SHSAT registration rate.  Two features in our dataset are most negatively correlated with registration rate: the percentage of students which are chronically absent, as well as the "community school" indicator.  The former is very much intuitive, but the latter fact may be an noteworthy learning from this analysis.  It's also somewhat interesting to see that a _higher_ student-to-teacher ratio correlates positively with registration rate.
# 
# #### Interesting correlations
# * Proficiency in both ELA and math is strongly correlated with strong attendance rate
# * Conversely, proficiency is strongly negatively correlated with low attendance schools (% chronically absent).  _It may be that improving attendance gives the most "bang for the buck" in terms of performance._
# * Math and ELA proficiency have high correlation: in other words, schools tend to have students proficient in both of them, rather than just one.
# * Both math and ELA proficiency are moderately correlated with pupil:teacher ratio: this is surprising, because we would expect proficiency to go down as the ratio goes up.  This may be because of peer group and competitive effects.
# * Collaborative teachers, effective school leadership, and trust percentage are highly correlated between each other.  It seems as if they occur together.  _But they don't seem to have any correlation with high registrations._
# * Community schools are negatively correlated with proficiency.  This is probably due to a hidden variable:  they are located in neighborhoods that are not conducive for student performance.  By definition, community schools are "designed to counter environmental factors that impede student achievement".  _This also is an opportunity to consider._
# * Class sizes in English, Math and Science are all highly correlated.  It seems like we can use one to represent all.

# ## Correlation Matrix: demographic features
# 
# As the lack of diversity in the SHSAT registrations is part of our original problem statement, it is also interesting to look at the existing correlation between demographic-related features and test registrations.

# In[18]:


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
# 
# #### Interesting correlations
# 
# * Schools with high number of Hispanics also tend to have high economic need index
# * Schools with high number of Hispanics also tend to have high English learners
# * On the contrary, schools with high number of whites have low economic need index
# * Schools with high number of SHSAT registrations tend to have high percentage of Asians
# * Schools with high number of SHSAT registrations tend to have low economic need index
# * Schools with high number of Hispanics tend not to have Asians, indicating segregation
# * Schools with high number of economically disadvantaged students also tend to have Asians.  This is unusual and needs more investigation.
# * Schools with high number of blacks tend not to have whites or Hispanics, again indicating segregation
# * Schools tend to have high performers in both ELA and math, not just one of them.
# 
# The last row shows where we need to focus, based on demographics:
# * Schools with high economic need index
# * Schools with high percentage of blacks or Hispanics

# ## Correlation Matrix: location and economic features
# 
# We would also like to look at the correlation between high registration rate and the boroughs and economic variables.

# In[19]:


# choose key features for correlation matrix
geo_econ_features =  ['in_bronx',
                      'in_brooklyn',
                      'in_manhattan',
                      'in_queens',
                      'in_staten',
                      'economic_need_index',
                      'sie_provided',
                      'school_income_estimate',
                      'grade_7_ela_4s_economically_disadvantaged',
                      'grade_7_math_4s_economically_disadvantaged',
                      'high_registrations']

draw_heatmap(Xy_train[geo_econ_features], 'plots/corr_matrix_boroughs.png')


# We observe from this correlation matrix that the demographic and economics features in general are not strongly correlated with a high SHSAT registration rate.  There are very slight positive correlations for students in the Bronx or Brooklyn, as well as with the economic need index.  There is a slight negative correlation with SHSAT registration rate for students in Queens.  As such, we expect it will be unlikely that the indicator variables for the boroughs will have strong predictive value.
# 
# We also see a negative correlation between school income estimate and high SHSAT registration rate, but caution that we only have these income estimates for about 1/3 of the schools in our dataset.  Contrary to our hypothesis that the existence or non-existence of such a school income estimate for a particular school might be meaningful, the binary flag for whether a school provided an income estimate seems uncorrelated with registration rate.

# ### Outlier Analysis
# 
# Let us look at the distribution of test taker percentage across NYC schools.

# In[20]:


plt.rcParams['figure.figsize'] = [12, 3]
sns.boxplot(x=train_data['pct_test_takers'])
plt.show()


# There are quite a few outliers there.  Let us look at them.

# In[21]:


display(train_data[train_data['pct_test_takers'] >= 90].sort_values('pct_test_takers', ascending=False))


# ### NYC "Gifted & Talented" Program
# 
# We tried to analyze why these schools have such high enrollments.  When we did some research on the Internet, we found that New York actually has a "Gifted & Talented" Test that kids can take.  High-performers can go into "G&T Schools".  It is these schools that are the outliers above.
# 
# * The Anderson School: A "Gifted and Talented" school: "has an advanced math program, with fast-paced instruction, which encourages students to discover new approaches to math problems". [Source](https://www.testingmom.com/tests/gifted-talented-nyc/schools/).
# 
# * Mamie Fay: Has a "gifted and talented" [program](https://insideschools.org/school/30Q122).
# 
# * Janice Marie Knight: This school seems to be a clear outlier: it has 93% black student population, yet it has 97% SHSAT registration.  Turns out it has a gifted program called [SOAR](https://insideschools.org/school/18K235).
# 
# * Tag Young Scholars: The school explicitly identifies itself for [excellence](http://tagscholars.com/index.php/mission-statement/).
# 
# * G&T Citywide (30th Ave): The name says it all: "Gifted and Talented".
# 
# * Brooklyn School of Inquiry: [Gifted](https://insideschools.org/school/20K686) school.
# 
# * Booker T. Washington: Very [selective in admissions](https://insideschools.org/school/03M054).
# 
# * Christa Mcauliffe: Strong [special education program](https://insideschools.org/school/20K187).
# 
# * New York City Lab Middle School: [Very selective](https://insideschools.org/school/02M312).
