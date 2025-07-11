import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import math


FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.33


def get_passing_plays(year, week_start, week_end):
    print('getting passing plays')

    # Get all passing plays
    passing_play_data = get_play_data(year=year, pass_only=True)
    # print(len(passing_play_data))
    # print(passing_play_data.head())

    # Get tracking data of passing plays
    tracking_data = get_tracking_data(year=year, week_start=week_start, week_end=week_end)
    #TODO: Normalize the direction of all tracking data
    # filtered_tracking_data = []
    # for week_df in tracking_data:
    #     merged = week_df.merge(passing_play_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')
    #     filtered_tracking_data.append(merged)

    filtered_tracking_data = filter_tracking_data(tracking_data, passing_play_data)

    filtered_norm_tracking_data = normalize_field_direction(filtered_tracking_data)

    return passing_play_data, filtered_norm_tracking_data


def filter_tracking_data(tracking_data, passing_play_data):
    filtered_tracking_data = []

    # Filter out all tracking data for plays that are not included in 'passing_play_data'
    for week_df in tracking_data:
        merged = week_df.merge(passing_play_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')
        filtered_tracking_data.append(merged)

    return filtered_tracking_data


def normalize_field_direction(tracking_data, normalize_direction=True):
    normalized_tracking_data = []

    # Flip spatial features so that offense is always going from left-to-right
    for week_df in tracking_data:
        if normalize_direction:
            left_mask = week_df['playDirection'] == 'left'
            week_df.loc[left_mask, 'x'] = FIELD_LENGTH - week_df.loc[left_mask, 'x']
            week_df.loc[left_mask, 'o'] = (180 - week_df.loc[left_mask, 'o']) % 360
            week_df.loc[left_mask, 'dir'] = (180 - week_df.loc[left_mask, 'dir']) % 360
            week_df.loc[left_mask, 'playDirection'] = 'right_norm'

        # merged = week_df.merge(passing_play_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')
        normalized_tracking_data.append(week_df)

    return normalized_tracking_data


# For all passing plays, obtain all frames for each play beginning at "ball_snap" and ending on "pass_forward"/"qb_sack"/"qb_strip_sack"/"run"
def get_pocket_frames(play_data: DataFrame, tracking_data):
    play_tracking_dict = {}

    for i,row in play_data.iterrows():
        game_id = row['gameId']
        play_id = row['playId']

        print(f'searching for {game_id} - {play_id}')

        # Look through all weeks of tracking data for specific play
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

def get_player_data(year):
    file_path = f'/Volumes/T7/Machine_Learning/Datasets/NFL/misc/players_{year}.csv'
    data = pd.read_csv(file_path)
    return data



# Input: 11 defenders, Output: defenders that are rushers
# Method: Defenders that are within 3-4 yards of the LoS that move towards the QB in the first ~1.5 seconds
def detect_rushers(all_def_players, tracking_data, ball_coord, qb_coord):
    print('PROCESSING RUSHERS')

    ball_x, ball_y = ball_coord
    qb_x, qb_y = qb_coord

    # Observe difference in defenders positions/velocity from the snap and 1.5 seconds later
    time_delay = 15 # 1.5 sec * 10 frames/sec
    start_frame = tracking_data['frameId'].min()
    end_frame = min(tracking_data['frameId'].max(), start_frame+time_delay)
    print(f'start:{start_frame}, end:{end_frame}')

    # Filter play tracking data to only include defenders
    frame_defenders = tracking_data[tracking_data['nflId'].isin(all_def_players)]
    print(frame_defenders)

    # Get defender tracking data at the ball snap
    starting_positions = frame_defenders[frame_defenders['frameId'] == start_frame]
    print('starting_positions:\n', starting_positions)

    # Get defender tracking data 1.5 seconds after ball snap
    ending_positions = frame_defenders[frame_defenders['frameId'] == end_frame]
    print('ending_positions:\n', ending_positions)

    los_dist_thresh = 5.0
    close_to_los = starting_positions[np.abs(starting_positions['x'] - ball_x) <= los_dist_thresh]
    print('close_to_los:\n', close_to_los)


# Input: RBs, TEs, and FBs, Output: players that are blocking
# Method: players that remain relatively close to their starting coords after ~1.5 seconds, that have a low velocity
def detect_blockers():
    offense_block_positions = ['RB', 'FB', 'TE']
    pass


def scale_player_coordinates(player_x, player_y, x_scale=128/FIELD_LENGTH, y_scale=64/FIELD_WIDTH):
    x_scaled = player_x * x_scale
    y_scaled = player_y * y_scale
    return x_scaled, y_scaled






def plot_frame(df, frame_id, title):
    frame = df[df['frameId'] == frame_id]
    plt.figure(figsize=(12, 6.5))

    teams = df['club'].unique().tolist()
    teams.remove('football')
    print('TEAMS:', teams)
    plt.scatter(frame['x'], frame['y'], c=frame['club'].map({teams[0]: 'blue', teams[1]: 'red', 'football': 'black'}), s=60)

    for _, row in frame.iterrows():
        plt.text(row['x']+0.5, row['y'], 'ball' if math.isnan(row['jerseyNumber']) else int(row['jerseyNumber']), fontsize=8)

    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.title(title)
    plt.xlabel('X (Field Length)')
    plt.ylabel('Y (Sideline to Sideline)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.savefig(f'{title}.png')

    