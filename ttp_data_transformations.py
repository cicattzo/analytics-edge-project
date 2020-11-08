import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import collections

def check_data_consistency(df_train, df_valid, df_test, categorical_features):
    error_counter = 0
    for var in categorical_features:
        if all(
            x in df_train[var].unique() for x in df_valid[var].unique()
        ) and all(
            x in df_train[var].unique() for x in df_test[var].unique()
        ):
            continue
        else:
            print('problem', var)
            error_counter += 1
            
    if error_counter==0:
        print("data is good, we can work")
        return(True)
    else:
        print('data is bad, change something in categorical variables')
        return(False)


def train_valid_trainvalid_test_split(df, categorical_features):
    df_test = df[df['open_dt_year']==2019][df['closed_dt_year']==2019]
    df_valid = df[df['open_dt_year']==2018][df['closed_dt_year']==2018]
    df_train = df[df['open_dt_year']<2018][df['closed_dt_year']<2018]
    df_trainvalid = df[df['open_dt_year']<2019][df['closed_dt_year']<2019]

    df_test = df_test.drop(['open_dt_year','closed_dt_year'], axis = 1)
    df_valid = df_valid.drop(['open_dt_year','closed_dt_year'], axis = 1)
    df_train = df_train.drop(['open_dt_year','closed_dt_year'], axis = 1)
    df_trainvalid = df_trainvalid.drop(['open_dt_year','closed_dt_year'], axis = 1)
    
    data_coll = collections.namedtuple('data', ['df_train', 'df_valid', 'df_trainvalid', 'df_test'])
    
    data_processed = data_coll(df_train, df_valid, df_trainvalid, df_test)
    
    if check_data_consistency(df_train, df_valid, df_test, categorical_features):
        return data_processed
    else:
        print("it doesnt work, do something")
        return

def X_y_identification(df_train, df_valid, df_trainvalid, df_test, categorical_features):
    x_train = df_train.drop(['actual_time','current_predicted_time'], axis = 1)
    y_train = df_train['actual_time'].values
    x_valid = df_valid.drop(['actual_time','current_predicted_time'], axis = 1)
    y_valid = df_valid['actual_time'].values
    x_test = df_test.drop(['actual_time','current_predicted_time'], axis = 1)
    y_current_prediction = df_test['current_predicted_time'].values
    y_test = df_test['actual_time'].values
    x_trainvalid = df_trainvalid.drop(['actual_time','current_predicted_time'], axis = 1)
    y_trainvalid = df_trainvalid['actual_time'].values

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    x_preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)])
    
    x_train_processed = x_preprocessor.fit_transform(x_train)
    x_valid_processed = x_preprocessor.transform(x_valid)
    x_trainvalid_processed = x_preprocessor.transform(x_trainvalid)    
    x_test_processed = x_preprocessor.transform(x_test)
    
    data_coll = collections.namedtuple('data', ['x_train', 'x_valid', 'x_trainvalid', 'x_test',
                                                'y_train', 'y_valid', 'y_trainvalid', 'y_test',
                                                'x_train_processed', 'x_valid_processed',
                                                'x_trainvalid_processed', 'x_test_processed',])
    data_processed = data_coll(x_train, x_valid, x_trainvalid, x_test,
                               y_train, y_valid, y_trainvalid, y_test,
                               x_train_processed.toarray(), x_valid_processed.toarray(),
                               x_trainvalid_processed.toarray(), x_test_processed.toarray())
    
    return data_processed

def data_transformations(df, categorical_features):
    print("start dataframe train validation trainvalidationsplit\n")
    data_processed = train_valid_trainvalid_test_split(df, 
                                                       categorical_features)
    
    print("start X y splits and preprocessing")
    data_processed_xy_identification = X_y_identification(data_processed.df_train, 
                                        data_processed.df_valid, 
                                        data_processed.df_trainvalid, 
                                        data_processed.df_test, 
                                        categorical_features)
    
    return data_processed_xy_identification
    
    
    
    
                                        
                                        
                                        
                                        
                                        
                                        
                                        
    
