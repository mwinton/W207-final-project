
# coding: utf-8

# # SHSAT Test Results Merge Notebook
# [Return to project overview](final_project_overview.ipynb)
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)
# 
# In this notebook, we will merge the data cleaned by the other "prep_" notebooks to create a single merged csv.

# ## Importing dataframes, indexed by our primary key
# While school names may change or be input inconsistently, each school has a unique identifying DBN, sometimes referred to as a Location Code, to identify it. By importing each cleaned dataset with the DBN as the index, we are able to easily join them into a merged dataset.

# In[29]:


import pandas as pd
import datetime
import re

# set default options
pd.set_option('display.max_columns', None)


# In[44]:


# Load all datasets from CSV; when loading set index to the DBN column (to enforce uniqueness)
shsat_df = pd.read_csv('data_cleaned/cleaned_shsat_outcomes.csv', index_col="dbn")
print('SHSAT dataset:',shsat_df.shape) # confirm that it's (589, 5)

class_sizes_df = pd.read_csv('data_cleaned/cleaned_class_sizes.csv', index_col="dbn")
print('Class size dataset:', class_sizes_df.shape) # confirm that it's (494, 13)

explorer_df = pd.read_csv('data_cleaned/cleaned_explorer.csv', index_col="dbn")
print('Explorer dataset:', explorer_df.shape) # confirm that it's (596, 55)

selectiveness_df = pd.read_csv('data_cleaned/selectiveness.csv', index_col='dbn')
print('Selectiveness dataset:', selectiveness_df.shape) # confirm that it's (589, 2)


# ## Checking for duplicate entries.
# We do a quick check to make sure there are no duplicate entries.

# In[31]:


shsat_dups = shsat_df.index.duplicated()
class_sizes_dups = class_sizes_df.index.duplicated()
explorer_dups = explorer_df.index.duplicated()
selectiveness_dups = selectiveness_df.index.duplicated()

print("True or False: there are duplicated indices within any dataframes?")
print("{0}.".format(bool(sum(shsat_dups) + sum(class_sizes_dups) + sum(explorer_dups) + 
                         sum(selectiveness_dups))))


# ## Inner joins for more complete data
# We'll use inner joins to select the intersection of our datasets, thus only selecting for schools for which we have data from each dataframe.

# In[32]:


merged_df = shsat_df.join(explorer_df, how="inner")
merged_df = merged_df.join(class_sizes_df, how="inner")
merged_df = merged_df.join(selectiveness_df, how="inner")
print("Merged Dataframe shape:",merged_df.shape)


# In[33]:


merged_df.head()


# In[34]:


print("Merged DF shape:",merged_df.shape)


# ## Evaluating density
# Let's take a look at how sparse our data is.

# In[35]:


print("Total empty cells:",merged_df.isnull().sum().sum())
print("Percent null: {0:.3f}%".format(100*merged_df.isnull().sum().sum()/(merged_df.shape[0]*merged_df.shape[1])))


# Let's take a look at our worst offending rows and columns to see if anything stands out enough to be removed:
# 
# ### Columns with Nulls

# In[36]:


merged_df.isnull().sum()[merged_df.isnull().sum() > 0]    .sort_values(ascending=False)


# ### Rows with Nulls

# In[37]:


merged_df.isnull().sum(axis=1)[merged_df.isnull().sum(axis=1) > 0]    .sort_values(ascending=False)


# At the moment we don't see any of these as being offending enough to be removed, especially since we have already preserved some info from the 'school_income_estimate' feature.

# ## Save a dated file
# 
# To allow updates to the merged dataframe without disrupting work on models downstream until they are ready, we save a dated merged filename.

# In[38]:


# Get the date to create the filename.
d = datetime.date
filename = "combined_data_{0}.csv".format( d.today().isoformat() )
print(filename)


# In[39]:


# check final shape (464,69)
merged_df.shape


# In[40]:


merged_df.to_csv("data_merged/{0}".format(filename))


# ## Save alternate dataset without class size information
# Because we are missing class size data for approximately 100 schools, the `inner join` used to merge our dataframes drops those rows.  We will also save a variant of our dataset without the class size data, in case it turns out those features have low predictve value in our models.

# In[41]:


no_class_size_df = shsat_df.join(explorer_df, how="inner")
no_class_size_df = no_class_size_df.join(selectiveness_df, how="inner")
print("Merged Dataframe shape (without class size data):",no_class_size_df.shape)


# ### Verify that characteristics of the dataset (in terms of nulls) are similar to above

# In[42]:


print("Total empty cells:",no_class_size_df.isnull().sum().sum())
print("Percent null: {0:.3f}%".format(100*no_class_size_df.isnull().sum().sum()/
                                      (no_class_size_df.shape[0]*no_class_size_df.shape[1])))

# check columns with nulls
no_class_size_df.isnull().sum()[no_class_size_df.isnull().sum() > 0]    .sort_values(ascending=False)


# There characteristics are similar to our primary dataset, so we should feel comfortable using it if we do not need the class size data in our models.  Note that several of the columns with nulls in our primary merged dataset originally came from the class size data.  As a result, aside from `school_income_estimate`, our columns look quite good with respect to nulls.

# In[43]:


# Get the date to create the filename.
filename = "combined_data_no_class_sizes_{0}.csv".format( d.today().isoformat() )
print(filename)

# check final shape (556, 62)
print(no_class_size_df.shape)

no_class_size_df.to_csv("data_merged/{0}".format(filename))

