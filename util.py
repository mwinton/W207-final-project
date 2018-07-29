# Utility functions

import re
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split

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

def read_data(data_file='data_merged/combined_data_2018-07-27.csv'):
    merged_df = pd.read_csv(data_file)
    X = merged_df.loc[:, ~merged_df.columns.isin(['high_registrations'])]
    y = merged_df.loc[:, merged_df.columns.isin(['high_registrations'])]
    train_data, test_data, train_labels, test_labels = our_train_test_split(X, y)
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
