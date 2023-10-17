# list of imports

#standard ds imports
import numpy as np
import pandas as pd

#stats and plotting
from pydataset import data
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

# core imports
import env
import os



#---------------------------------------------------------------------------#
# variables




#---------------------------------------------------------------------------#
#Functions

def get_db_url(db, user=env.user, host=env.host, password=env.password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#---------------------------------------------------------------------------#

def new_titanic_data(sql_query):
    """
    This function will:
    - take in a sql_query
    - create a connection url
    - return a df of the given query from the databse  
    """
    # create the connection url:
    url = get_db_url('titanic_db')
    
    return pd.read_sql(sql_query, url)



def get_titanic_data(filename="titanic.csv"):
    """
    This function will:
    - Check loacal directory for csv file
        - return df if file exists
        - If csv doesn't exist:
            - create a df of the sql query
            - write the df to a csv file
    - output titanic df
    """
    
    directory = (f'{os.getcwd()}/')
    
    # checks if csv exists
    if os.path.exists(directory + filename):
        #if YES
        df = pd.read_csv(filename)
        return df
    #if NO
    else:
        #obtaning new data from sql
        df = new_titanic_data()
        
        #convert to a csv
        df.to_csv(filename)
        return df 

    
#---------------------------------------------------------------------------#

# This function will create the dataframe. 
def new_iris_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a connection_url to mySQL
    - return a df of the given query from the iris_db
    """
    url = get_db_url('iris_db')
    
    return pd.read_sql(SQL_query, url)
    

# This function will save the dataframe into a csv. If on already exists it will pull the old csv. 

def get_iris_data(directory = (f'{os.getcwd()}/') , filename = 'iris.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv
    - outputs iris df
    """
    
    #directory = (f'{os.getcwd()}/')
    
    #checks if csv exists
    if os.path.exists(directory+filename): 
        # if Yes
        df = pd.read_csv(filename)
        return df
    #if NO
    else:
        #obtaining new data from sql
        df = new_iris_data(SQL_query)
        #covert to a csv
        df.to_csv(filename)
        return df
    
#---------------------------------------------------------------------------# 
    
    
def new_telco_data(sql_query):
    """
    This function will:
    - take in a sql_query
    - create a connection url
    - return a df of the given query from the databse  
    """
    # create the connection url:
    url = get_db_url('telco_churn')
    return pd.read_sql(sql_query, url) 
    
    
def get_telco_data(filename="telco_churn.csv"):
    """
    This function will:
    - take in an SQL_query
    - Check loacal directory for csv file
        - return df if file exists
        - If csv doesn't exist:
            - create a df of the sql query
            - write the df to a csv file
    - output telcho churn df
    """
    directory = (f'{os.getcwd()}/')
    
    # checks if csv exists
    if os.path.exists(directory + filename):
        #if YES
        df = pd.read_csv(filename)
        return df
    #if NO
    else:
        #obtaning new data from sql
        df = new_telco_data()
        
        #convert to a csv
        df.to_csv(filename)
        return df      
    