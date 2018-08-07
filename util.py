# Utility functions

import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
from functools import partial
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold


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


def get_dummies(train_data, test_data, factor_cols=['zip','district']):
    '''
        inputs: train_data, test_data (pandas dataframes)
        returns: train_data_ohe, test_data_ohe (pandas dataframes)
        NOTE: any factors discovered in test set, which weren't in training, are ignored
    '''
    
    for f in factor_cols:
        train_dummies = pd.get_dummies(train_data[f], prefix=f)
        tmp_test_dummies = pd.get_dummies(test_data[f], prefix=f)

        # create empty test_dummies dataframe; fill only with columns also in train_dummies
        # it's important to have the same number of columns in both dataframes
        test_dummies = pd.DataFrame()
        
        train_dummy_cols = train_dummies.columns.values
        tmp_test_dummy_cols = tmp_test_dummies.columns.values
        
        # this loop ensures the same dummy columns in train & test (in the same order)
        for col in train_dummy_cols:
            # if a training dummy was also created for test, keep the test dummy
            if col in tmp_test_dummy_cols:
                test_dummies[col]=tmp_test_dummies[col]
            # if training a dummy was not created for test, add it (with all zeros)
            else:
                test_dummies[col] = pd.Series([0 for x in range(len(tmp_test_dummies.index))],
                                              index=tmp_test_dummies.index)
    
        # join the new dummies to the original dataset
        train_data_merged = train_data.join(train_dummies, how="inner")
        test_data_merged = test_data.join(test_dummies, how="inner")

    # drop original columns after one hot encoding
    train_data_ohe = train_data_merged.drop(factor_cols, axis=1)
    test_data_ohe = test_data_merged.drop(factor_cols, axis=1)
    
    print('Train data initial shape:',train_data.shape)
    print('Test  data initial shape:',test_data.shape)
    print('Train data OHE\'d shape:',train_data_ohe.shape)
    print('Test  data OHE\'d shape:',test_data_ohe.shape)

    return train_data_ohe, test_data_ohe
    
def get_num_pcas (train_data, var_explained=0.9):
    # Determine the number of principal components to achieve target explained variance
    cum_explained_variance_ratios = [0]

    # default number of PCA to number of features
    n_pca = train_data.shape[1]
    for n in range(1, 21):
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=n))
        pipeline.fit_transform(train_data)
        pca = pipeline.steps[1][1]
        cum_explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))
        # stop once we've hit the target variance explained
        if (np.sum(pca.explained_variance_ratio_) >= var_explained):
            # store n_pca for future use
            print ('With %d principal components, variance explained = %.3f.' %
                   (n, np.sum(pca.explained_variance_ratio_)))
            n_pca = n
            break

    plt.plot(np.array(cum_explained_variance_ratios))
    plt.xlabel('# Principal Components')
    plt.ylabel('% Variance Explained')
    plt.grid()
    plt.show()

    return n_pca

def ohe_data(train_data, test_data, factor_cols=['zip','district']):
    '''
        DEPRECATED!!!! Use util.get_dummies(...) instead.
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
    

def read_data(data_file='data_merged/combined_data_2018-07-30.csv', do_imputation=False):

    merged_df = pd.read_csv(data_file)

    if do_imputation:
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
    else:
        imputed_df = merged_df

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


def run_model_get_ordered_predictions(pipeline,
                                      train_orig, test_orig,
                                      train_best, test_best,
                                      train_labels, test_labels):
    """Function that runs a model and returns results that represent
    an ordering over schools that are most likely to have high registrations
    and schools that are least likely to have high registrations."""
    # recombine train and test data into an aggregate dataset
    X_orig = pd.concat([train_orig, test_orig], sort=True)  # including all columns (need for display purposes)
    X_best = pd.concat([train_best, test_best], sort=True)  # only columns from best model
    y = np.concatenate((train_labels, test_labels))

    # Run k-fold cross-validation with 5 folds 10 times, which means every school is predicted 10 times.
    folds = 5
    repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=207)

    # Build dataframes for storing predictions, with columns for each k-fold
    fold_list = []
    for f in range(1, (folds * repeats) + 1):
        fold_list.append('k{}'.format(f))
    predictions = pd.DataFrame(index=X_best.index, columns=fold_list)

    counter = 1
    for train, test in rskf.split(X_best, y):
        pipeline.fit(X_best.iloc[train, ], y[train, ])
        predicted_labels = pipeline.predict(X_best.iloc[test, ])
        predictions.iloc[test, counter - 1] = predicted_labels
        counter += 1

    # Create columns for predictions and labels
    predictions['1s'] = predictions.iloc[:, :50].sum(axis=1)
    predictions['0s'] = (predictions.iloc[:, :50] == 0).sum(axis=1)
    predictions['true'] = y

    # Create a table of all features along with the number of votes each received and the true value
    X_predicted = pd.concat([X_orig, predictions['1s'], predictions['0s'],
                             pd.DataFrame(y)], axis=1, join_axes=[X_best.index])
    # Rename the y columns to be more descriptive
    X_predicted.rename(columns={0: "high_registrations"}, inplace=True)
    # Sort by number of votes, most votes for 1 at the top and most votes for 0 at the bottom
    X_predicted = X_predicted.sort_values(by=['1s', '0s'], ascending=[False, True])

    return X_predicted


def create_passnyc_list(X_predicted, train_orig, test_orig, train_labels, test_labels):
    """Function that takes the false positives and returns a rank order dataframe"""

    # Retrain only the columns of interest for PASSNYC prioritization
    fp_features = ['1s',
                  'dbn',
                  'school_name',
                  'economic_need_index',
                  'grade_7_enrollment',
                  'num_shsat_test_takers',
                  'pct_test_takers',
                  'percent_black__hispanic'
                  ]
    df_passnyc = X_predicted[fp_features]

    X_orig = pd.concat([train_orig, test_orig], sort=True)  # including all columns (need for display purposes)
    y = np.concatenate((train_labels, test_labels))

    # Determine the number of test takers this school would have needed to meet
    #   the median percentage of high_registrations
    median_pct = np.median(X_orig[y == 1]['pct_test_takers']) / 100
    target_test_takers = np.multiply(df_passnyc['grade_7_enrollment'], median_pct)

    # Subtract the number of actual test takers from the hypothetical minimum number
    delta = target_test_takers - df_passnyc['num_shsat_test_takers']

    # Multiply the delta by the minority percentage of the school to determine how many minority
    #   students did not take the test
    minority_delta = np.round(np.multiply(delta, df_passnyc['percent_black__hispanic'] / 100), 0).astype(int)
    df_passnyc = df_passnyc.assign(minority_delta=minority_delta)

    # Multiply the minority delta by the percentage of 1 votes to get a confidence-adjusted 'score'
    df_passnyc = df_passnyc.assign(score=np.multiply(df_passnyc['minority_delta'], df_passnyc['1s'] / 10))

    # sort descending and create a rank order column
    df_passnyc = df_passnyc.sort_values(by='score', ascending=False)
    df_passnyc.insert(0, 'rank', range(1, df_passnyc.shape[0] + 1))

    return df_passnyc

