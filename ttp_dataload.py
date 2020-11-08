import pandas as pd

def download_data():
    #download dataset
    df = pd.read_csv("./data/clean_311_service_requests_2019_to_2015.csv")
    #drop na in target_dt
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df