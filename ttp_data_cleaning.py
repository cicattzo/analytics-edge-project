import pandas as pd
import numpy as np


def date_preprocessing(df):
    # dates from str to datetime
    df["open_dt"] = pd.to_datetime(df["open_dt"])
    df["target_dt"] = pd.to_datetime(df["target_dt"])
    df["closed_dt"] = pd.to_datetime(df["closed_dt"])
    #actual spent time
    df["actual_time"] = (df["closed_dt"] - df["open_dt"])
    df["actual_time"] = df['actual_time'] / np.timedelta64(1, 'h')

    df["current_predicted_time"] = (df["target_dt"] - df["open_dt"])
    df["current_predicted_time"] = df['current_predicted_time'] / np.timedelta64(1, 'h')
    
    return df
    
def column_selection(df):
    #all case_status = closed
    df = df.drop(['case_status'], axis = 1)
    #drop id - case_enquiry_id
    df = df.drop(['case_enquiry_id'], axis = 1)
    #drop target_dt - we care about actual time, current target time prediction will just add bias, 
    # because we want to predict better than current approahc
    df = df.drop(['target_dt'], axis = 1)
    # drop ontime, because that is our final target
    df = df.drop(['ontime'], axis = 1)
    # drop location_street_name because of too high granularity
    df = df.drop(['location_street_name'], axis = 1)
    #drop case_title - Most titles are self evident. This field is entered by call takers on a call by call basis
    df = df.drop(['case_title'], axis = 1)
    #drop type because depending on year some types are not presented so our model will not know what to do with some values of the column
    df = df.drop(['type'], axis = 1)
    #the same thing with location_zipcode
    df = df.drop(['location_zipcode'], axis = 1)

    #for now lets drop precinct, i think that thing superconnected with location
    df = df.drop(['precinct'], axis = 1)
    
    return df

def date_feature_engineering(df):
    #lets work with datetimes
    df['open_dt_year'] = df['open_dt'].dt.year
    df['open_dt_month'] = df['open_dt'].dt.month
    df['open_dt_week'] = df['open_dt'].dt.week
    df['open_dt_day'] = df['open_dt'].dt.day
    df['open_dt_hour'] = df['open_dt'].dt.hour
    # df['open_dt_minute'] = df['open_dt'].dt.minute
    df['open_dt_dayofweek'] = df['open_dt'].dt.dayofweek

    df['closed_dt_year'] = df['closed_dt'].dt.year
    df['closed_dt_month'] = df['closed_dt'].dt.month
    df['closed_dt_week'] = df['closed_dt'].dt.week
    df['closed_dt_day'] = df['closed_dt'].dt.day
    df['closed_dt_hour'] = df['closed_dt'].dt.hour
    # df['closed_dt_minute'] = df['closed_dt'].dt.minute
    df['closed_dt_dayofweek'] = df['closed_dt'].dt.dayofweek

    df = df.drop(['open_dt'], axis = 1)
    df = df.drop(['closed_dt'], axis = 1)
    
    return df
    
def full_cleaning(df):
    print("data preprocessing started\n")
    df = date_preprocessing(df)
    print("column selection started\n")
    df = column_selection(df)
    print("feature engineering started\n")
    df = date_feature_engineering(df)
    return df
    