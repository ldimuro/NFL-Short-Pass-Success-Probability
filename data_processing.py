import pandas as pd
from pandas import DataFrame
import numpy as np

def get_passing_plays(year, week_start, week_end):
    print('getting passing plays')

    tracking_data = get_tracking_data(year=year, week_start=week_start, week_end=week_end)

    passing_play_data = get_play_data(year=year, pass_only=True)
    print(len(passing_play_data))
    print(passing_play_data.head())

    filtered_tracking_data = []
    for week_df in tracking_data:
        merged = week_df.merge(passing_play_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')
        filtered_tracking_data.append(merged)

    return passing_play_data, filtered_tracking_data


# For all passing plays, obtain all frames for each play beginning at "ball_snap" and ending on "pass_forward"/"qb_sack"/"qb_strip_sack"/"run"
def get_pocket_frames(play_data: DataFrame, tracking_data):
    play_tracking_dict = {}

    for i,row in play_data.iterrows():
        game_id = row['gameId']
        play_id = row['playId']

        print(f'searching for {game_id} - {play_id}')

        # Look through all weeks of tracking data
        tracking_play = None
        for i,week_df in enumerate(tracking_data):
            match = week_df[(week_df['gameId'] == game_id) & (week_df['playId'] == play_id)]
            
            if not match.empty:
                tracking_play = match.sort_values('frameId')
                print('FOUND in week', i+1)
                break

        if tracking_play is not None:

            # Remove all frames before 'ball_snap' and after QB passes, gets sacked, or scrambles
            start_event = 'ball_snap'
            end_events = ['pass_forward', 'qb_sack', 'qb_strip_sack', 'run']
            snap_index = tracking_play[tracking_play['event'] == start_event].index
            end_index = tracking_play[tracking_play['event'].isin(end_events)].index

            if not snap_index.empty and not end_index.empty:
                tracking_play = tracking_play.loc[snap_index[0]:end_index[-1]].reset_index(drop=True)
            else:
                print("🚨Warning: No 'ball_snap' found in play")

            # Add all relevant tracking frames to plays dict
            play_tracking_dict[(game_id, play_id)] = tracking_play
            print(f'processed tracking frames for {game_id} - {play_id}')
        else:
            print(f'🚨could not find {game_id} - {play_id}')

    return play_tracking_dict
    



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

    