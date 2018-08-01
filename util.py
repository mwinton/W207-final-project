# Utility functions

import re
from functools import partial
from sklearn.model_selection import train_test_split

### Cleanup utility functions

def pct_to_number(df, col, type=int):
    """Pass type=float if you have floating point values"""
    return df[col].astype(str).apply(lambda s: type(s.strip('%')))
def money_to_number(df, col):
    """Pass type=float if you have floating point values"""
    return df[col].astype(str).apply(lambda s: type(re.sub('[$,]', '', s)))
    
# Having spaces etc. can cause annoying problems: replace with underscores
def sanitize_column_names(c):
    c = c.lower()
    c = re.sub('[?,()/]', '', c)
    c = re.sub('[ -]', '_', c)
    c = c.replace('%', 'percent')
    return c

### Define our data split
TEST_SIZE = 0.2
RANDOM_STATE = 207
our_train_test_split = partial(train_test_split, test_size=TEST_SIZE, random_state=RANDOM_STATE)

