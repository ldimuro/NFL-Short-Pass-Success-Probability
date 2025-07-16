import pandas as pd
from pandas import DataFrame
import numpy as np
import constants


def filter_tracking_data(tracking_data, passing_play_data):
    filtered_tracking_data = []

    # Filter out all tracking data for plays that are not included in 'passing_play_data'
    for week_df in tracking_data:
        merged = week_df.merge(passing_play_data[['gameId', 'playId']], on=['gameId', 'playId'], how='inner')
        filtered_tracking_data.append(merged)

    return filtered_tracking_data


def normalize_field_direction(tracking_data):
    normalized_tracking_data = []

    # Flip spatial features so that offense is always going from left-to-right
    for week_df in tracking_data:
        week_df = week_df.copy()

        left_mask = week_df['playDirection'] == 'left'
        week_df.loc[left_mask, 'x'] = constants.FIELD_LENGTH - week_df.loc[left_mask, 'x']
        week_df.loc[left_mask, 'o'] = (180 - week_df.loc[left_mask, 'o']) % 360
        week_df.loc[left_mask, 'dir'] = (180 - week_df.loc[left_mask, 'dir']) % 360
        week_df.loc[left_mask, 'playDirection'] = 'right_norm'

        normalized_tracking_data.append(week_df)

    return normalized_tracking_data

def normalize_to_center(play_frames, ball_coord):
    normalized_play_frames = play_frames.copy()
    ball_x, ball_y = ball_coord

    # Place ball x-axis at center field, and adjust all other players' positions around it
    shift_x = constants.CENTER_FIELD - ball_x
    normalized_play_frames['x'] += shift_x

    return normalized_play_frames




def get_relevant_frames(play_data: DataFrame, tracking_data, start_events, end_events):
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

        # Remove all frames before start_events and after end_events
        if tracking_play is not None:

            # Plays will either start with 'huddle_break_offense' or NA
            if 'START' in start_events:
                start_index = tracking_play[(tracking_play['event'].isin(['huddle_break_offense'])) | (tracking_play['event'].isna())].index
            else:
                start_index = tracking_play[tracking_play['event'].isin(start_events)].index

            # Plays will end with NA
            if 'END' in end_events:
                end_index = tracking_play[tracking_play['event'].isna()].index
            else:
                end_index = tracking_play[tracking_play['event'].isin(end_events)].index

            if not start_index.empty and not end_index.empty:
                tracking_play = tracking_play.loc[start_index[0]:end_index[-1]].reset_index(drop=True)
            else:
                print(f"🚨Warning: Missing event in play")

            # Add all relevant tracking frames to plays dict
            play_tracking_dict[(game_id, play_id)] = tracking_play
            print(f'processed tracking frames for {game_id} - {play_id}')
        else:
            print(f'🚨could not find {game_id} - {play_id}')

    return play_tracking_dict
    


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


def scale_player_coordinates(player_x, player_y, x_scale=128/constants.FIELD_LENGTH, y_scale=64/constants.FIELD_WIDTH):
    x_scaled = player_x * x_scale
    y_scaled = player_y * y_scale
    return x_scaled, y_scaled


