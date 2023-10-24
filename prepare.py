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

# function to clean
def clean_iris(df):
    '''
    This function will take in an iris.csv database.
    - it will drop column: species id
    - rename species_name to species
    - create dummy columns
    '''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


# function to split
def split_iris_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on species.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on species to get an even mix of each species
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    
    # splits train_validate into train and validate using train_test_split() stratifying on species to get an even mix of each species
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
# combined split and clean
def prep_iris(df):
    '''
    This function will:
    - drop column: species_id
    - rename species_name to species
    - create dummy columns
    - split the data into train, validate, and test
    '''
    
    # drop column using .drop(columns=column_name)
    df = df.drop(columns='species_id')
    
    # remame column using .rename(columns={current_column_name : replacement_column_name})
    df = df.rename(columns={'species_name':'species'})
    
    # create dummies dataframe using .get_dummies(column_name,not dropping any of the dummy columns)
    dummy_df = pd.get_dummies(df['species'], drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df], join along the index)
    df = pd.concat([df, dummy_df], axis=1)
    
    # split data into train/validate/test using split_data function
    train, validate, test = split_iris_data(df)
    
    return train, validate, test

#prep iris 2
def prep_iris_2(iris) -> pd.DataFrame:
    '''
    prep_iris will take a single positional argument,
    a single pandas DataFrame,
    and will output a cleaned version of the dataframe
    this is expected to receive the data output by 
    get_iris_data from acquire module, see documentation
    for acquire.py for further details
    return: pd.DataFrame
    '''
    # drop that species_id column:
    iris = iris.drop(columns='species_id')
    # rename that species_name column into species for cleanliness:
    iris = iris.rename(columns={'species_name':'species'})
    return iris

#----------------------------------------------------------------------------------------------#
# Titanic-data:

def clean_titanic_data(df):
    '''
    This function will clean the data prior to splitting.
    '''
    # Drops any duplicate values
    df = df.drop_duplicates()

    # Drops columns that are already represented by other columns
    cols_to_drop = ['deck', 'embarked', 'class', 'passenger_id']
    df = df.drop(columns=cols_to_drop)

    # Fills the small number of null values for embark_town with the mode
    df['embark_town'] = df.embark_town.fillna(value='Southampton')

    # Uses one-hot encoding to create dummies of string columns for future modeling 
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)

    return df

def split_titanic_data(df):
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    # Creates the test set
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify=df.survived)
    
    # Creates the final train and validate set
    train, validate = train_test_split(train, test_size=.3, random_state=123, stratify=train.survived)
    
    return train, validate, test

# age is missing numbers: function to add age mean

def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test

# combination of clean, split, and impute mean age:

def prep_titanic_data(df):
    '''
    Combines the clean_titanic_data, split_titanic_data, and impute_mean_age functions.
    '''
    df = clean_titanic_data(df)
    df = df.drop(columns='sex')
    df = df.drop(columns='embark_town')

    train, validate, test = split_titanic_data(df)
    
    train, validate, test = impute_mean_age(train, validate, test)

    return train, validate, test



# another prep
def prep_titanic(titanic) -> pd.DataFrame:
    '''
    prep_titanic will take in a single pandas DataFrame, titanic
    as expected from the acquire.py return of get_titanic_data
    it will return a single cleaned pandas dataframe
    of this titanic data, ready for analysis.
    '''
    titanic = titanic.drop(columns=[
        'Unnamed: 0',
        'passenger_id',
        'embarked',
        'deck',
        'class'
    ])
    titanic.loc[:,'age'] = titanic.age.fillna(round(titanic.age.mean())).values
    titanic.loc[:, 'embark_town'] = titanic.embark_town.fillna('Southampton')
    return titanic

#--------------------------------------------------------------------------------------#
#Telco Data:

def split_telco_data(df):
    '''
    This function performs split on telco data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=1349, 
                                        stratify=df.Churn)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=1349, 
                                   stratify=train_validate.Churn)
    return train, validate, test



def prep_telco(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop(columns=['InternetService', 'Contract', 'PaymentMethod'])
    df['gender'] = df.gender.map({'Female': 1, 'Male': 0})
    df['Partner'] = df.Partner.map({'Yes': 1, 'No': 0})
    df['Dependents'] = df.Dependents.map({'Yes': 1, 'No': 0})
    df['PhoneService'] = df.PhoneService.map({'Yes': 1, 'No': 0})
    df['PaperlessBilling'] = df.PaperlessBilling.map({'Yes': 1, 'No': 0})
    df.TotalCharges.fillna('20.20',inplace=True)
    df['Churn'] = df.Churn.map({'Yes': 1, 'No': 0})
    dummy_df = pd.get_dummies(df[['MultipleLines', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']],
                              drop_first=True)
    df = df.drop(columns=['MultipleLines', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test



def prep_telco_2(telco) -> pd.DataFrame:
    '''
    prep_telco will take in a a single pandas dataframe
    presumed of the same structure as presented from 
    the acquire module's get_telco_data function (refer to acquire docs)
    returns a single pandas dataframe with redudant columns
    removed and missing values filled.
    '''
    telco = telco.drop(
    columns=[
        'Unnamed: 0',
        'internet_service_type_id',
        'payment_type_id',
        'contract_type_id',   
    ])
    telco.loc[:,'internet_service_type'] = telco.internet_service_type.\
    fillna('no internet')
    telco = telco.set_index('customer_id')
    telco.loc[:,'total_charges'] = (telco.total_charges + '0')
    telco.total_charges = telco.total_charges.astype(float)
    return telco


#-----------------------------------------------------------------------------------------#
#spliting data:

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
    