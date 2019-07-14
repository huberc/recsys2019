"""
Exmple usage in notebook:


import pandas as pd
import SessionItemMatrix as sim 

dataframe = <split dataset from trivago>

dataframe = sim.prepare_dataset(dataframe)

(R, (session_id_labels, session_id_levels), (reference_labels, reference_levels)) = sim.sparse_matrix(dataframe)
"""

import csv
import pandas as pd
import numpy as np
from scipy import sparse as sp

def remove_rows_with_non_numeric_reference(dataframe):
    """
    removes rows with non-numeric references
    return: dataframe with numeric references only
    """
    dataframe.dropna(subset=['reference'], inplace=True)
    # https://stackoverflow.com/questions/33961028/remove-non-numeric-rows-in-one-column-with-pandas
    return dataframe[dataframe.reference.apply(lambda x: x.isnumeric())]

def give_weights(dataframe):
    """
    gives weights based on action_type
    return: dataframe with weight column added
    """
    one_hot_action_type = pd.get_dummies(dataframe.action_type)
    # dataframe['weight'] = np.zeros(len(dataframe.reference), np.int8) 
    dataframe['weight'] = one_hot_action_type['interaction item image'] * 1.0 + \
        one_hot_action_type['search for item'] * 1.0 + \
        one_hot_action_type['interaction item rating'] * 1.0 + \
        one_hot_action_type['interaction item deals'] * 1.0 + \
        one_hot_action_type['interaction item info'] * 1.0 + \
        one_hot_action_type['clickout item'] * 2.0
    return dataframe

def drop_columns(dataframe):
    """
    drops columns which are not used anymore
    return: dataframe with only session_id, reference and weight columns
    """
    return dataframe.drop(['user_id', 'action_type', 'timestamp', 'step', 'platform', 'city', 'device', 'current_filters', 'impressions', 'prices'], axis=1)

def group_by(dataframe):
    """
    group by session and reference in order to sum up the weights
    """
    return dataframe.groupby(['session_id','reference'], as_index=False).sum()

def prepare_dataset(dataframe):
    """
    removes rows with non-numeric references,
    gives weights based on action_type,
    drops columns which are not used anymore
    group by session and reference
    return: the prepared dataframe
    """
    dataframe = remove_rows_with_non_numeric_reference(dataframe)
    dataframe = give_weights(dataframe)
    dataframe = drop_columns(dataframe)
    dataframe = group_by(dataframe)
    return dataframe

def sparse_matrix(prepared_dataset):
    """
    generates labels and levels by factorizing session_id and reference
    return: sparse matrix and labels, levels of session_id and reference
    """
    session_id_labels, session_id_levels = pd.factorize(prepared_dataset['session_id'])
    reference_labels, reference_levels = pd.factorize(prepared_dataset['reference'])
    R = sp.csr_matrix((prepared_dataset.weight, (session_id_labels, reference_labels)))
    return (R, (session_id_labels, session_id_levels), (reference_labels, reference_levels))