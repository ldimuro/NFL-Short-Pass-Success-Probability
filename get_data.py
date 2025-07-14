import pandas as pd
import data_processing

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

def get_player_data(year):
    file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/misc/players_{year}.csv'
    data = pd.read_csv(file_path)
    return data

def get_passing_plays(year, week_start, week_end):
    print('getting passing plays')

    # Get all passing plays
    passing_play_data = get_play_data(year=year, pass_only=True)

    # Get tracking data of all plays
    tracking_data = get_tracking_data(year=year, week_start=week_start, week_end=week_end)

    # Get only tracking data of passing plays
    filtered_tracking_data = data_processing.filter_tracking_data(tracking_data, passing_play_data)

    # Flip spatial features to always have offense going left-to-right
    filtered_norm_tracking_data = data_processing.normalize_field_direction(filtered_tracking_data)

    return passing_play_data, filtered_norm_tracking_data