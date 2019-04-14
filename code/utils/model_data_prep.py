from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pprint
import os,sys

sys.path.insert(0,os.getcwd()+'/code/utils/')
from toolkit import load_pickle

def normalize(df, with_std=False):
    # fill na
    df = df.fillna(0)
    # scale data. output: ndarray
    scaler = StandardScaler(with_std=with_std)

    score_scale = scaler.fit_transform(df)
    # turn scaled data into df.
    df_scale = pd.DataFrame(score_scale, index=df.index, columns=df.columns)

    return df_scale

def weighted_avg(criteria_scores,region_list, debug=None):
    """
    for the purpose of evaluating one model one region model(ARIMA/LSTM)
    :param criteria_scores: a list of list of mae/mse/etc scores.
    :return:
    """

    # weighted average
    weights = load_pickle("region_weights.p", path='output/')


    # get ordered subset of weight value
    weight_list = [weights[k] for k in region_list if k in weights]


    if debug:
        weight_list = list(weights.values())[: debug]

    pprint.pprint(weight_list)
    weighted_scores = []
    for score_list in criteria_scores:
        weighted_scores.append(np.average(score_list, weights=weight_list))
    return weighted_scores