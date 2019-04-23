# https: // pythonprogramming.net/static/downloads/machine-learning-data/crypto_data.zip

import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np

# Constants
SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    '''Function which give the target value regarding the future price'''
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    '''Preprocessing function:
        * Normalize values
        * Drop NaNs
        * Get sequential_data set
    '''
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    # make sure both lists are only up to the shortest length.
    buys = buys[:lower]
    # make sure both lists are only up to the shortest length.
    sells = sells[:lower]

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data) # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array! ..import numpy as np

# Joins all csv files into one dataframe
main_df = pd.DataFrame()  # begin empty

# the 4 ratios we want to consider
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:  # begin iteration
    #print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    # read in specific file
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close",
                       "volume": f"{ratio}_volume"}, inplace=True)

    # set time as index so we can join them on this shared time
    df.set_index("time", inplace=True)
    # ignore the other columns besides price and volume
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

# if there are gaps in data, use previously known values
main_df.fillna(method="ffill", inplace=True)
main_df.dropna(inplace=True)
#print(main_df.head())


# Adds target column to dataframe
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))


# Out of sample dataset
times = sorted(main_df.index.values)  # get the times
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times
#print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the last 5%

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
