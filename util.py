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
PASSNYC_LABELS = [0,1]
our_train_test_split = partial(train_test_split,
                               stratify=PASSNYC_LABELS,
                               test_size=TEST_SIZE,
                               random_state=RANDOM_STATE)



