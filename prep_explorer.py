
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

# In[12]:


# import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# set default options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

get_ipython().magic('matplotlib inline')


# In[13]:


# load dataset from CSV
raw_se_2016 = pd.read_csv('data_raw/2016_school_explorer.csv')
raw_se_2016.info()


# In[14]:


raw_se_2016.head()


# ## Trimming
# There is considerably more here than we need, starting with the fact that the dataset includes primary schools that only go up to 5th grade.  We trim the dataset to only those schools that have a 7th grade.

# In[15]:


se_2016_trimmed = raw_se_2016[raw_se_2016['Grades'].str.contains('07')]
se_2016_trimmed.info()


# This cuts the number of entries roughly in half, but there are also many more columns than we need.  We want to keep the information that describes the school's location, practices and demographics, and the academic descriptors of the 7th grade (these students will take the SHSAT exam in the fall of their 8th grade).

# In[16]:


features_to_keep = ['Location Code', 'School Name', 'District','Zip', 'Community School?',
                    'Economic Need Index', 'School Income Estimate',
                    'Percent ELL', 'Percent Asian', 'Percent Black', 'Percent Hispanic',
                    'Percent Black / Hispanic', 'Percent White', 'Student Attendance Rate',
                    'Percent of Students Chronically Absent',
                    'Rigorous Instruction %','Rigorous Instruction Rating',
                    'Collaborative Teachers %','Collaborative Teachers Rating',
                    'Supportive Environment %','Supportive Environment Rating',
                    'Effective School Leadership %','Effective School Leadership Rating',
                    'Strong Family-Community Ties %','Strong Family-Community Ties Rating',
                    'Trust %', 'Trust Rating', 
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


# In[17]:


se_2016_trimmed.head()


# ## Derived Features
# We will now derive some features to group some data together or preserve some information from features we'll otherwise need to discard.

# In[18]:


print("Shape before derived columns:",se_2016_trimmed.shape)
se_2016_derived = se_2016_trimmed.copy()

# While we are missing too many values to use the income values
# in our 'School Income Estimate' feature, it may be that the
# administrative capacity or parental responsiveness to even be
# able to report such a figure may be indicative of some aspect
# of a school community that may be worth considering, so we note
# the presence or absence of those values.
se_2016_derived['sie_provided'] = -se_2016_derived['School Income Estimate'].isna()
se_2016_derived['sie_provided'] = se_2016_derived['sie_provided'].astype(int)

# While zip codes might be too granular on their own to signal
# commonalities between schools, we see if grouping by borough
# provides more shared information, and allow the 'district' variable
# to signal more fine-grained locality.
bronx_zips = [10453, 10457, 10460, 10458, 10467, 10468, 10451, 10452, 10456, 10454, 10455, 10459, 10474, 10463, 10471, 10466, 10469, 10470, 10475, 10461, 10462,10464, 10465, 10472, 10473]
brooklyn_zips = [11212, 11213, 11216, 11233, 11238, 11209, 11214, 11228, 11204, 11218, 11219, 11230, 11234, 11236, 11239, 11223, 11224, 11229, 11235, 11201, 11205, 11215, 11217, 11231, 11203, 11210, 11225, 11226, 11207, 11208, 11211, 11222, 11220, 11232, 11206, 11221, 11237]
manhattan_zips = [10026, 10027, 10030, 10037, 10039, 10001, 10011, 10018, 10019, 10020, 10036, 10029, 10035, 10010, 10016, 10017, 10022, 10012, 10013, 10014, 10004, 10005, 10006, 10007, 10038, 10280, 10002, 10003, 10009, 10021, 10028, 10044, 10065, 10075, 10128, 10023, 10024, 10025, 10031, 10032, 10033, 10034, 10040]
queens_zips = [11361, 11362, 11363, 11364, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11365, 11366, 11367, 11412, 11423, 11432, 11433, 11434, 11435, 11436, 11101, 11102, 11103, 11104, 11105, 11106, 11374, 11375, 11379, 11385, 11691, 11692, 11693, 11694, 11695, 11697, 11004, 11005, 11411, 11413, 11422, 11426, 11427, 11428, 11429, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11368, 11369, 11370, 11372, 11373, 11377, 11378]
staten_zips = [10302, 10303, 10310, 10306, 10307, 10308, 10309, 10312,10301, 10304, 10305,10314]

se_2016_derived['in_bronx'] = se_2016_derived['Zip'].apply((lambda zip: int(zip in bronx_zips)))
se_2016_derived['in_brooklyn'] = se_2016_derived['Zip'].apply((lambda zip: int(zip in brooklyn_zips)))
se_2016_derived['in_manhattan'] = se_2016_derived['Zip'].apply((lambda zip: int(zip in manhattan_zips)))
se_2016_derived['in_queens'] = se_2016_derived['Zip'].apply((lambda zip: int(zip in queens_zips)))
se_2016_derived['in_staten'] = se_2016_derived['Zip'].apply((lambda zip: int(zip in staten_zips)))

print("Shape after derived columns:",se_2016_derived.shape)


# ## Cleanup
# We will now define some utility functions to clean up some columns and column names for easier analysis. These changes include:
# 
# * Converting column names to lowercase
# * Stripping '%' and '$' symbols from data and column names
# * Converting the rating system from 'Approaching Target', 'Meeting Target', and 'Exceeding Target' to 1, 2, or 3
# * Drop the rating columns for which we have corresponding percent columns (there are more missing values in the rating columns)
# * Converting the Yes/No community schools string to binary

# In[19]:


# Apply changes to new dataframe
se_2016_renamed = se_2016_derived.copy()
se_2016_renamed.rename(columns={"Location Code":"DBN"}, inplace=True)


# In[20]:


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
    'Rigorous Instruction Rating',
    'Collaborative Teachers Rating',
    'Supportive Environment Rating',
    'Effective School Leadership Rating',
    'Strong Family-Community Ties Rating',
    'Trust Rating',
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


# In[21]:


plt.hist(se_2016_renamed.loc[se_2016_renamed['average_ela_proficiency']        .notnull(),'average_ela_proficiency'], bins=20)
plt.show()
plt.hist(se_2016_renamed.loc[se_2016_renamed['average_math_proficiency']        .notnull(),'average_math_proficiency'], bins=20)
plt.show()


# In[22]:


# check final shape (rows = number of schools)
# should be (596, 50)
se_2016_renamed.shape


# In[23]:


# save the cleaned dataset to CSV
se_2016_renamed.to_csv('data_cleaned/cleaned_explorer.csv', index=False)

