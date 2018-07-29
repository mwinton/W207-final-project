
# coding: utf-8

# # School Explorer Preliminary EDA / Cleaning Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)
# 
# The [2016 School Explorer](https://www.kaggle.com/passnyc/data-science-for-good#2016%20School%20Explorer.csv) dataset includes highly detailed information about all 1200+ schools in the five boroughs of New York.  This information includes:
# 
# * Physical information like grade, latitude/longitude, district, location and SED codes, etc.
# * Descriptive information such as Grades, whether it is a community school
# * Financial information such as Budget, Economic Need Index
# * Demographic information such as percent Asian, Black, etc., percent ELL
# * School program information such as Rigorous Instruction, Collaborative Teachers, Supportive Environment, etc.
# * Academic achievement information in the form of per-grade and per-economic/demographic breakdowns of number of Math/ELA students who receive scores of '4' (highest score)
# 

# In[1]:


# import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().magic('matplotlib inline')


# In[2]:


# load dataset from CSV
raw_se_2016 = pd.read_csv('data_raw/2016_school_explorer.csv')
raw_se_2016.info()


# In[3]:


raw_se_2016.head()


# ## Trimming
# There is considerably more here than we need, starting with the fact that the dataset includes primary schools that only go up to 5th grade.  We trim the dataset to only those schools that have a 7th grade.

# In[4]:


se_2016_trimmed = raw_se_2016[raw_se_2016['Grades'].str.contains('07')]
se_2016_trimmed.info()


# This cuts the number of entries roughly in half, but there are also many more columns than we need.  We want to keep the information that describes the school's location, practices and demographics, and the academic descriptors of the 7th grade (these students will take the SHSAT exam in the fall of their 8th grade).

# In[5]:


features_to_keep = ['Location Code', 'School Name', 'District','Zip', 'Community School?', 'Economic Need Index', 'School Income Estimate',
                    'Percent ELL', 'Percent Asian', 'Percent Black', 'Percent Hispanic',
                    'Percent Black / Hispanic', 'Percent White', 'Student Attendance Rate',
                    'Percent of Students Chronically Absent', 'Rigorous Instruction %',
                    'Collaborative Teachers %',
                    'Supportive Environment %',
                    'Effective School Leadership %',
                    'Strong Family-Community Ties %',
                    'Trust %', 
                    'Student Achievement Rating', 'Average ELA Proficiency',
                    'Average Math Proficiency', 
                    'Grade 7 ELA - All Students Tested',
                    'Grade 7 ELA 4s - All Students',
                    'Grade 7 ELA 4s - American Indian or Alaska Native',
                    'Grade 7 ELA 4s - Black or African American',
                    'Grade 7 ELA 4s - Hispanic or Latino',
                    'Grade 7 ELA 4s - Asian or Pacific Islander',
                    'Grade 7 ELA 4s - White', 'Grade 7 ELA 4s - Multiracial',
                    'Grade 7 ELA 4s - Limited English Proficient',
                    'Grade 7 ELA 4s - Economically Disadvantaged',
                    'Grade 7 Math - All Students Tested', 
                    'Grade 7 Math 4s - All Students',
                    'Grade 7 Math 4s - American Indian or Alaska Native',
                    'Grade 7 Math 4s - Black or African American',
                    'Grade 7 Math 4s - Hispanic or Latino',
                    'Grade 7 Math 4s - Asian or Pacific Islander',
                    'Grade 7 Math 4s - White',
                    'Grade 7 Math 4s - Multiracial',
                    'Grade 7 Math 4s - Limited English Proficient',
                    'Grade 7 Math 4s - Economically Disadvantaged'
]
se_2016_trimmed = se_2016_trimmed[features_to_keep]
se_2016_trimmed.info()


# In[6]:


se_2016_trimmed.head()


# ## Cleanup
# We will now define some utility functions to clean up some columns and column names for easier analysis. These changes include:
# 
# * Converting column names to lowercase
# * Stripping '%' and '$' symbols from data and column names
# * Converting the rating system from 'Approaching Target', 'Meeting Target', and 'Exceeding Target' to 1, 2, or 3
# * Drop the rating columns for which we have corresponding percent columns (there are more missing values in the rating columns)
# * Converting the Yes/No community schools string to binary

# In[7]:


# Apply changes to new dataframe
se_2016_renamed = se_2016_trimmed.copy()
se_2016_renamed.rename(columns={"Location Code":"DBN"}, inplace=True)


# In[8]:


# Utility functions 
import util

# Remove percent
percent_columns = [
    'Percent ELL',
    'Percent Asian',
    'Percent Black',
    'Percent Hispanic',
    'Percent Black / Hispanic',
    'Percent White',
    'Student Attendance Rate',
    'Percent of Students Chronically Absent',
    'Rigorous Instruction %',
    'Collaborative Teachers %',
    'Supportive Environment %',
    'Effective School Leadership %',
    'Strong Family-Community Ties %',
    'Trust %'
]

# Remove dollar signs
money_columns = [
    'School Income Estimate'
]

# Convert ratings to numeric
rating_columns = [
#     'Rigorous Instruction Rating',
#     'Collaborative Teachers Rating',
#     'Supportive Environment Rating',
#     'Effective School Leadership Rating',
#     'Strong Family-Community Ties Rating',
#     'Trust Rating',
    'Student Achievement Rating'
]

# Convert Yes/No to to 1/0
binary_columns= [
    'Community School?'
]

for col in percent_columns:
    se_2016_renamed[col] = util.pct_to_number(se_2016_renamed, col)
for col in money_columns:
    se_2016_renamed[col] = util.money_to_number(se_2016_renamed, col)
for col in rating_columns:
    se_2016_renamed[col] = util.rating_to_number(se_2016_renamed, col)
for col in binary_columns:
    se_2016_renamed[col] = util.to_binary(se_2016_renamed, col)

se_2016_renamed.columns = [util.sanitize_column_names(c) for c in se_2016_renamed.columns]
se_2016_renamed.head()


# In[9]:


plt.hist(se_2016_renamed.loc[se_2016_renamed['average_ela_proficiency']        .notnull(),'average_ela_proficiency'], bins=20)
plt.show()
plt.hist(se_2016_renamed.loc[se_2016_renamed['average_math_proficiency']        .notnull(),'average_math_proficiency'], bins=20)
plt.show()


# In[10]:


# check final shape (rows = number of schools)
# should be (596, 50)
se_2016_renamed.shape


# In[11]:


# save the cleaned dataset to CSV
se_2016_renamed.to_csv('data_cleaned/cleaned_explorer.csv', index=False)

