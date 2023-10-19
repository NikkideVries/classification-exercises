# imports

import numpy as np
import pandas as pd
import env
import acquire

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer




#----------------------------------------------------------------------------------------------#
#Variables





#----------------------------------------------------------------------------------------------#
# Iris Data:

# clean_iris data:

def clean_iris(df):
    """
    clean_iris will take an acquired df and 
    remove `species_id` and `measurement_id` columns and 
    rename `species_name` column to just `species` and
    encode 'species_name' column into TWO new columns
    
    return: single cleaned dataframe
    """
    
    # drops the columns we don't want
    dropcols = ['species_id']
    df.drop(columns= dropcols, inplace=True)
    # renames the species_name to species
    df.rename(columns={'species_name': 'species'}, inplace=True)
    # creates a dummy column: assign numbered values to species
    dummy_sp = pd.get_dummies(df[['species']], drop_first=True)

    # returns the new dataframe with the dummy table attached
    return pd.concat([df, dummy_sp], axis =1)



def prep_iris(df):
    """
    prep_iris will take one argument(df) and 
    run clean_iris to remove/rename/encode columns
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """
    iris_df = clean_iris(df)
    train_validate, test = train_test_split(iris_df, test_size=0.2, random_state=3210, stratify=iris_df.species)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210, stratify=train_validate.species)
    return train, validate, test




def split_data(df, dataset=None):
    target_cols = {
        'telco': 'churn',
        'titanic': 'survived',
        'iris': 'species'
    }
    if dataset:
        if dataset not in target_cols.keys():
            print('please choose a real dataset tho')

        else:
            target = target_cols[dataset]
            train_val, test = train_test_split(
                df,
                train_size=0.8,
                stratify=df[target],
                random_state=1349)
            train, val = train_test_split(
                train_val,
                train_size=0.7,
                stratify=train_val[target],
                random_state=1349)
            return train, val, test
    else:
        print('please specify what df we are splitting.')
    