# Utility functions

import re
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder


### Cleanup utility functions

def pct_to_number(df, col, type=int):
    """Pass type=float if you have floating point values"""
    return df[col].astype(str).apply(lambda s: int(s.strip('%')) if s != 'nan' else np.nan)

def money_to_number(df, col, type=float):
    """Pass type=float if you have floating point values"""
    return df[col].astype(str).apply(lambda s: type(re.sub('[$,]', '', s)) if s != 'nan' else np.nan)

def translate_ratings(rating):
    if rating == "Exceeding Target":
        return 4
    elif rating == "Meeting Target":
        return 3
    elif rating == "Approaching Target":
        return 2
    elif rating == "Not Meeting Target":
        return 1
    elif s != 'nan':
        return 0

def rating_to_number(df, col, type=int):
    return df[col].astype(str).apply(lambda s: translate_ratings(s) if s != 'nan' else np.nan)

def to_binary(df, col, type=int):
    return df[col].astype(str).apply(lambda s: 1 if s=='Yes' else 0)

# Having spaces etc. can cause annoying problems: replace with underscores
def sanitize_column_names(c):
    c = c.lower()
    c = re.sub('[?,()/]', '', c)
    c = re.sub('\s-\s', '_', c)
    c = re.sub('[ -]', '_', c)
    c = c.replace('%', 'percent')
    return c

### Define our data split
TEST_SIZE = 0.2
RANDOM_STATE = 207
our_train_test_split = partial(train_test_split,
                               test_size=TEST_SIZE,
                               random_state=RANDOM_STATE)

def ohe_data(train_data, test_data, factor_cols=['zip','district']):
    '''
        inputs: train_data, test_data (pandas dataframes)
        returns: train_data_ohe, test_data_ohe (both sparse matrices)
    '''
    
    # get indices for specified columns
    factor_col_ids = []
    for f in factor_cols:
        idx = train_data.columns.get_loc(f) # column order assumed to be same in test set
        factor_col_ids.append(idx)
    factor_col_ids = np.array(factor_col_ids)

    # perform one hot encoding; return full matrix (not sparse) for compatibility with PCA
    ohe_enc = OneHotEncoder(categorical_features=factor_col_ids, sparse=False, handle_unknown='ignore')
    train_data_ohe = ohe_enc.fit_transform(train_data)
    test_data_ohe = ohe_enc.transform(test_data)
    print('Train data initial shape:',train_data.shape)
    print('Test  data initial shape:',test_data.shape)
    print('Train data new shape:',train_data_ohe.shape)
    print('Test  data new shape:',test_data_ohe.shape)
    
    return train_data_ohe, test_data_ohe
    
def read_data(data_file='data_merged/combined_data_2018-07-28.csv'):
    merged_df = pd.read_csv(data_file)
    
    # these columns cannot/should not be imputed
    # notably, don't impute for `school_income_estimate` because too many missing values
    non_impute_cols = ['dbn', 
                       'school_name',
                       'district',
                       'zip',
                       'school_income_estimate']
    
    # temporarily split out the non-numeric cols into a separate dataframe
    tmp_non_numeric_df = merged_df[non_impute_cols]
    tmp_numeric_df = merged_df.drop(non_impute_cols, axis=1)
    
    # do imputation of missing values to column mean
    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    tmp_imputed_df = pd.DataFrame(imp.fit_transform(tmp_numeric_df))
    tmp_imputed_df.columns = tmp_numeric_df.columns
    tmp_imputed_df.index = tmp_numeric_df.index
    
    # reassemble into a single dataframe
    imputed_df = pd.concat([tmp_non_numeric_df, tmp_imputed_df], axis=1)

    # split into features (X) and labels (y)
    X = imputed_df.loc[:, ~imputed_df.columns.isin(['high_registrations'])]
    y = imputed_df.loc[:, imputed_df.columns.isin(['high_registrations'])]
    train_data, test_data, train_labels, test_labels = our_train_test_split(X, y, stratify=y)
    
    # convert y values into 1D array, as expected by sklearn classifiers
    train_labels = train_labels.values.ravel()
    test_labels = test_labels.values.ravel()
    
    # confirm stratification
    print('Train: %d observations (positive class fraction: %.3f)' %
          (len(train_labels), np.sum(train_labels==1) / len(train_labels)))
    print('Test : %d observations (positive class fraction: %.3f)' % 
          (len(test_labels), np.sum(test_labels==1) / len(test_labels)))

    return train_data, test_data, train_labels, test_labels

def print_cv_results(cv_scores):
    k_folds = len(cv_scores['test_accuracy'])	# any of them will do

    # display accuracy with 95% confidence interval
    cv_accuracy = cv_scores['test_accuracy']
    print('With %d-fold cross-validation, accuracy is: %.3f (95%% CI from %.3f to %.3f).' %
          (k_folds, cv_accuracy.mean(), cv_accuracy.mean() - 1.96 * cv_accuracy.std(),
           cv_accuracy.mean() + 1.96 * cv_accuracy.std()))

    # display F1 score with 95% confidence interval
    cv_f1 = cv_scores['test_f1']
    print('The F1 score is: %.3f (95%% CI from %.3f to %.3f).' %
          (cv_f1.mean(), cv_f1.mean() - 1.96 * cv_f1.std(),
           cv_f1.mean() + 1.96 * cv_f1.std()))
