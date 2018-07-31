
# coding: utf-8

# # Using machine learning to prioritize NYC middle schools for intervention, with an intent to increase diverse student registrations for the specialized high school admissions test
# 
# ### Andrew Larimer, Deepak Nagaraj, Daniel Olmstead, Michael Winton (W207-4-Summer 2018 Final Project)
# 

# Paraphrased from the Kaggle [PASSNYC: Data Science for Good Challenge](https://www.kaggle.com/passnyc/data-science-for-good) overview (June 2018):
# 
# ## Problem Statement
# 
# PASSNYC is a not-for-profit organization dedicated to broadening educational opportunities for New York City's talented and underserved students. In recent years, the City’s specialized high schools - institutions with historically transformative impact on student outcomes - have seen a shift toward more homogeneous student body demographics.  PASSNYC aims to increase the diversity of students taking the Specialized High School Admissions Test (SHSAT). By focusing efforts in underperforming areas that are historically underrepresented in SHSAT registration, we will help pave the path to specialized high schools for a more diverse group of students.
# 
# PASSNYC and its partners provide outreach services that improve the chances of students taking the SHSAT and receiving placements in these specialized high schools. The current process of identifying schools is effective, but PASSNYC could have an even greater impact with a more informed, granular approach to quantifying the potential for outreach at a given school. Proxies that have been good indicators of these types of schools include data on English Language Learners, Students with Disabilities, Students on Free/Reduced Lunch, and Students with Temporary Housing.
# 
# Part of this challenge is to assess the needs of students by using publicly available data to quantify the challenges they face in taking the SHSAT. The best solutions will enable PASSNYC to identify the schools where minority and underserved students stand to gain the most from services like after school programs, test preparation, mentoring, or resources for parents.
# 
# More on [PASSNYC](http://www.passnyc.org/opportunity-explorer/).
# 

# ## Overview of our approach
# 
# We are going to approach this analysis as a classification problem, by defining a class label related to the proportion of students at a school who register for the SHSAT test (ie. we will define "high-registration" schools).  We will build models with multiple machine learning algorithms such as LogisticRegression, K-Nearest Neighbors, Random Forests, and Neural Networks.  We will explore both feature-rich models, as well as models built after feature selection or a dimensionality reduction technique such as PCA, have been used to reduce the number of features in the model.  
# 
# For assessing quality of our models, we will use an 80/20 train/test split of our dataset.  We will use K-fold cross-validation within the training set for hyperparameter optimization, and will calculate average accuracy and F1 scores (along with 95% confidence intervals) based on the cross-validation results.  Once hyperparameter optimization is complete, we will run the model against our held-out test set and report accuracy and F1 score.
# 
# Following that, we will analyze the highest scored false positives from each model **TBD: with a sufficiently high F1 score?**, and prioritize them for PASSNYC engagement based on attributes such as % registrations, % black and Hispanic students, in alignment with the organization's mission.

# ## Datasets
# 
# We joined several datasets together for this project. We focused mainly on data from the 2016-2017 period as predictors for the 2017 SHSAT test, which is taken in the fall. 
# 
# ### 1. School Explorer 2016
# 
# The [2016 School Explorer](https://www.kaggle.com/passnyc/data-science-for-good#2016%20School%20Explorer.csv) dataset was provided by PASSNYC on Kaggle.  It contains key information about every school in NYC such as name, district, location (address and lat/long), grade levels, budget, demographics, and number of students who achieved 4's in Math and ELA testing, by various demographic and economic levels.  It also contains a set of "school quality metrics" such as rigorous instruction rating, collaborative teachers rating, and strong family-community ties rating.

# ### 2. NYC SHSAT Test Results 2017
# 
# The [NYC SHSAT Test Results 2017](https://www.kaggle.com/willkoehrsen/nyc-shsat-test-results-2017/home) dataset contains data from the New York Times Article: ["See Where New York City’s Elite High Schools Get Their Students"](https://www.nytimes.com/interactive/2018/06/29/nyregion/nyc-high-schools-middle-schools-shsat-students.html) by Jasmine Lee published June 29, 2018.  Data was parsed and uploaded to Kaggle by [Richard W DiSalvo](https://www.kaggle.com/rdisalv2).  This dataset contains information on schools with students eligible to take the SHSAT, the number of students who took the test, the number of resulting offers, and a basic demographic percentage of Black/Hispanic students at the school (ie, NOT test-takers).

# ### 3. NYC Class Size Report 2016-2017
# 
# The [2016-2017 NYC Class Size Report](https://www.kaggle.com/marcomarchetti/20162017-nyc-class-size-report) dataset originally came from the [NYC Schools website](http://schools.nyc.gov/AboutUs/schools/data/classsize/classsize_2017_2_15.htm), but is no longer available there.  It was parsed and uploaded to Kaggle by [Marco Marchetti](https://www.kaggle.com/marcomarchetti).  It is a merge of three datasets: "K-8 Avg, MS HS Avg, PTR".  The "MS HS Avg" subset gives the average class size by program, department, and subject for each school.  The "PTR" data gives the pupil-teacher ratio for the school.

# ### 4. Demographic Snapshot School 2013-2018
# 
# This [2013-2018 Demographic Snapshot of NYC Schools](https://data.cityofnewyork.us/Education/2013-2018-Demographic-Snapshot-School/s52a-8aq6) was downloaded directly from the NYC Open Data project. It contains grade-level enrollments for each school from the NY Department of Education.

# ### 5. Gifted and Talented Schools Lists
# 
# New York City also has a set of test-in Gifted & Talented (G&T) programs.  Some of city-wide, and others give preference to students within a particular district.  Data on which schools have these programs was scraped from [insideschools.org](http://insideschools.org).

# ## Preliminary EDA and Data Cleaning
# 
# Each of these datasets required varying degrees of cleaning before they could be joined together.  The EDA and data cleaning was done in separate notebooks, and results saved as CSV files.
# 
# 1. [School Explorer EDA Notebook](prep_explorer.ipynb)
# 2. [SHSAT Results & Demographic Snapshot Notebook](prep_shsat_results.ipynb)
# 3. [Class Size Notebook](prep_class_sizes.ipynb)
# 4. [Gifted & Talented Web Scraping Script](sel_scrape.py)
# 
# Next we load the resulting CSV files in a [Merge Notebook](prep_merge.ipynb) to join our cleaned data into one master dataset, resolve issues with missing values, and save as [combined_data.csv](data_merged/combined_data.csv).

# ## Highlights of Exploratory Data Analysis
# 
# **TODO: complete this section**

# In[1]:


from IPython.display import Image
Image(filename='plots/corr_matrix_key_features.png') 


# As discussed in our [EDA notebook](eda_correlation_matrics.ipynb), from this correlation matrix, we see that in general, feature related to test scores or academic proficiency are positively correlated with a high SHSAT registration rate.  Two features in our dataset are most negatively correlated with registration rate: the percentage of students which are chronically absent, as well as the "community school" indicator.  The former is very much intuitive, but the latter fact may be an noteworthy learning from this analysis.  It's also somewhat interesting to see that a _higher_ student-to-teacher ratio correlates positively with registration rate.

# In[2]:


Image(filename='plots/corr_matrix_demographics.png') 


# Also as previously mentioned in our EDA notebook, we see that percent Asian and percent white correlate positively with high SHSAT registration rates, whereas percent black and Hispanic correlate negatively with the registration rate.  This is in agreement with the original problem statement that PASSNYC posed.  Additionally, schools with a higher economic need index correlate with lower registration rates.  However, even in light of this fact, it is particularly interesting to note that schools with a high proportion of economically disadvantage students scoring 4's on their ELA and Math exams tend to have higher registration rates, as well.

# In[3]:


Image(filename='plots/corr_matrix_boroughs.png')


# As discussed in our EDA notebook, we saw minimal correlation between indicator variables for each of the 5 NYC boroughs with high SHSAT registration rates.

# ## Model Building
# 
# Next, we apply a variety of machine learning techniques to determine which provide the best classification results.  Each technique is performed in a separate notebook.  In each case, we also evaluate the effects of using PCA for dimensionality reduction.
# 
# 1. [K-Nearest Neighbors Model](model_knn.ipynb)
# 2. [Random Forests](model_forests.ipynb)
# 4. [Logistic Regression](model_logreg.ipynb)
# 4. [Neural Network](model_neuralnet.ipynb)

# ## Summary
# 
# Here we summarize the results from each of our models.  Keep in mind that because the class labels are defined as "Top 25th Percentile", any model that does not achieve > 75% accuracy is _no better than a hardcoded model that predicts the negative class naively for **every** observation_.  Since we performed 10-fold cross-validation on each of our models, we also report a 95% confidence interval for both the accuracy and the F1 scores.
# 
# Model | CV Accuracy | (95% CI) | CV F1 | (95% CI) | Test Set Accuracy | Test Set F1
# :---|:---:|:---:|:---:|:---:|:---:|:---:
# K-Nearest Neighbors (Full Model) | foo | (a,b) | foo | (a,b) | foo | foo
# K-Nearest Neighbors (Top n Features) | foo | (a,b) | foo | (a,b) | foo | foo
# Random Forest | foo | (a,b) | foo | (a,b) | foo | foo
# Logistic Regression | foo | (a,b) | foo | (a,b) | foo | foo
# Mulilayer Perceptron NN | foo | (a,b) | foo | (a,b) | foo | foo
# 
# From this **TBD: MODEL** appears to have the best classification accuracy on our dataset.  However, since ensembles in general have better performance than any individual model, we will use a combination of these models for predicting the specific schools we would recommend that PASSNYC engage with.

# ## Prioritized Engagement Recommendations
# 
# Since our dataset contains data for most NYC middle schools already, there is not a large separate, unlabeled dataset that we need to run through our model.  Instead, we will make our recommendations based on an analysis of false positives.  A false positive is a school which our models predicted should have a high registration rate, but did not.  We consider these to be the schools that are most likely to benefit from PASSNYC's intervention and engagement.
# 
# In analyzing these false positives, we chose to prioritize schools based on attributes that align with PASSNYC's goals:
# 
# 1. Larger student body - our assumption is that PASSNYC doesn't necessarily spend proportionally more resources to engage with a larger school.
# 2. Lower test registration rates - if schools just missed the threshhold, random variation could put them above the threshhold next year.
# 3. High % black and hispanic populations - this is directly aligned with the diversity objective of PASSNYC
# 4. **TBD: economic need?**
# 
# We first generated a list of the top 20 highest scoring false negatives from **TBD: which models were included**, applied this prioritization algorithm to each, and then took an ensemble vote from those lists.  The resulting schools are the ones that we recommend PASSNYC engage with for the highest anticipated ROI:

# In[ ]:


# display table of selected schools with the attributes used in prioritization


# ## Further Work
# 
# **TODO: decide whether we need anything here **
