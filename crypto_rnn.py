# https: // pythonprogramming.net/static/downloads/machine-learning-data/crypto_data.zip

import pandas as pd
import os
from sklearn import preprocessing

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
