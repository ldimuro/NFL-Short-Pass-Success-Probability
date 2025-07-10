import pandas as pd
import numpy as np

def get_passing_plays():
    print('getting passing plays')



def get_tracking_data(year, week_start, week_end):
    #TODO: Add dynamic way to obtain data if repo is copied from Github

    tracking_data = []

    for week in range(week_start, week_end+1):
        file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/tracking_data/{year}_tracking_week_{week}.csv'
        tracking_data.append(pd.read_csv(file_path))
        print(f'loaded {file_path}')

    return tracking_data

def get_play_data(year, pass_only=True):
    file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/misc/plays_{year}.csv'
    data = pd.read_csv(file_path)

    if pass_only:
        data = data[data['passResult'].notna()]

    return data

    