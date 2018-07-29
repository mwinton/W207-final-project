
# coding: utf-8

# # SHSAT Test Results Merge Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)
# 
# In this notebook, we will merge the data cleaned by the other "prep_" notebooks to create a single merged csv.

# ## Importing dataframes, indexed by our primary key
# While school names may change or be input inconsistently, each school has a unique identifying DBN, sometimes referred to as a Location Code, to identify it. By importing each cleaned dataset with the DBN as the index, we are able to easily join them into a merged dataset.

# In[1]:


import pandas as pd
import datetime
import re


# In[3]:


# Load all datasets from CSV; when loading set index to the DBN column (to enforce uniqueness)
shsat_df = pd.read_csv('data_cleaned/cleaned_shsat_outcomes.csv', index_col="dbn")
print('SHSAT dataset:',shsat_df.shape) # confirm that it's (589, 5)

class_sizes_df = pd.read_csv('data_cleaned/cleaned_class_sizes.csv', index_col="dbn")
print('Class size dataset:', class_sizes_df.shape) # confirm that it's (494,13)

explorer_df = pd.read_csv('data_cleaned/cleaned_explorer.csv', index_col="dbn")
print('Explorer dataset:', explorer_df.shape) # confirm that it's (596, 43)


# ## Checking for duplicate entries.
# We do a quick check to make sure there are no duplicate entries.

# In[4]:


shsat_dups = shsat_df.index.duplicated()
class_sizes_dups = class_sizes_df.index.duplicated()
explorer_dups = explorer_df.index.duplicated()
                            
print("True or False: there are duplicated indices within any dataframes?")
print("{0}.".format(bool(sum(shsat_dups) + sum(class_sizes_dups) + sum(explorer_dups))))


# ## Inner joins for more complete data
# We'll use inner joins to select the intersection of our datasets, thus only selecting for schools for which we have data from each dataframe.

# In[5]:


merged_df = shsat_df.join(explorer_df, how="inner")
merged_df = merged_df.join(class_sizes_df, how="inner")
print("Merged Dataframe shape:",merged_df.shape)


# In[6]:


merged_df.head()


# This still leaves us with a merged dataframe of 464 rows and 66 features.

# ## Evaluating density
# Let's take a look at how sparse our data is.

# In[7]:


print("Total empty cells:",merged_df.isnull().sum().sum())
print("Percent null: {0:.3f}%".format(100*merged_df.isnull().sum().sum()/(merged_df.shape[0]*merged_df.shape[1])))


# Let's take a look at our worst offending rows and columns to see if anything stands out enough to be removed:
# 
# ### Columns with Nulls

# In[8]:


merged_df.isnull().sum()[merged_df.isnull().sum() > 0]    .sort_values(ascending=False)


# ### Rows with Nulls

# In[9]:


merged_df.isnull().sum(axis=1)[merged_df.isnull().sum(axis=1) > 0]    .sort_values(ascending=False)


# At the moment we don't see any of these as being offending enough to be removed.

# ## Save a dated file
# 
# To allow updates to the merged dataframe without disrupting work on models downstream until they are ready, we save a dated merged filename.

# In[10]:


# Get the date to create the filename.
d = datetime.date
filename = "combined_data_{0}.csv".format( d.today().isoformat() )
print(filename)


# In[11]:


# check final shape (464,66)
merged_df.shape


# In[12]:


merged_df.to_csv("data_merged/{0}".format(filename))

