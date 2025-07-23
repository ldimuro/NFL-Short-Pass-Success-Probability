import pandas as pd
from pandas import DataFrame
import numpy as np
import constants
import math
import re
import pickle
from itertools import product


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
        week_df.loc[left_mask, 'o'] = (360 - week_df.loc[left_mask, 'o']) % 360
        week_df.loc[left_mask, 'dir'] = (360 - week_df.loc[left_mask, 'dir']) % 360
        week_df.loc[left_mask, 'playDirection'] = 'right_norm'

        normalized_tracking_data.append(week_df)

    return normalized_tracking_data

def normalize_to_center(tracking_data: DataFrame):
    normalized_weeks = []
    
    for week_df in tracking_data:
        week_df = week_df.copy()
        normalized_plays = []

        # Group by each play (gameId + playId)
        for (game_id, play_id), play_df in week_df.groupby(['gameId', 'playId']):
            ball_rows = play_df[play_df['team' if 'team' in week_df.columns else 'club'] == 'football']
            if ball_rows.empty:
                normalized_plays.append(play_df)
                continue

            # Calculate shift to move ball x to 60
            ball_x = ball_rows.iloc[0]['x']
            shift_x = 60 - ball_x

            play_df['x'] = play_df['x'] + shift_x
            normalized_plays.append(play_df)

        # Combine all normalized plays back into one DataFrame for the week
        normalized_weeks.append(pd.concat(normalized_plays, ignore_index=True))

    return normalized_weeks



def get_relevant_frames(play_data: DataFrame, tracking_data, start_events, end_events, extra_frames=0):
    play_tracking_dict = {}

    # Collapse all weeks of tracking data into 1 DataFrame
    tracking_data = pd.concat(tracking_data, ignore_index=True)

    for i,row in play_data.iterrows():
        game_id = row['gameId']
        play_id = row['playId']

        print(f'searching for {game_id} - {play_id}')

        # Look through all weeks of tracking data for specific play
        tracking_play = None
        match = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
        
        if not match.empty:
            tracking_play = match.sort_values('frameId')
            # print('FOUND in week', i+1)
            # break
        # for i,week_df in enumerate(tracking_data):
        #     match = week_df[(week_df['gameId'] == game_id) & (week_df['playId'] == play_id)]
            
        #     if not match.empty:
        #         tracking_play = match.sort_values('frameId')
        #         # print('FOUND in week', i+1)
        #         break

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
                print(f"🚨Warning: Missing event in play ({game_id},{play_id})")

            # Add all relevant tracking frames to plays dict
            play_tracking_dict[(game_id, play_id)] = tracking_play
            # print(f'processed tracking frames for {game_id} - {play_id}')
        else:
            print(f'🚨could not find {game_id} - {play_id}')

    return play_tracking_dict



def get_data_at_pass_forward(play_data: DataFrame, tracking_data: DataFrame, player_data):
    # Consolidate all tracking_weeks together
    tracking_data = pd.concat(tracking_data, ignore_index=True)

    # Extract first and last name of receiver of a play in the playDescription and create 2 new columns for the name
    play_data[['receiver_first_initial', 'receiver_last_name']] = play_data['playDescription'].apply(lambda desc: pd.Series(extract_first_and_last_name(desc)))

    # For each passing play:
    # 1) use first and last name to obtain the nflId of the receiver
    # 2) extract the tracking data at the moment of the pass for all 22 players on the field
    # 3) store the receiver_id and tracking data in a dictionary (with (gameId, playId) as the key)
    candidate_plays = {}
    skipped = 0
    for i,play in play_data.iterrows():
        game_id = play['gameId']
        play_id = play['playId']
        play_df = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
        los = np.round(play_df[play_df['team'] == 'football'].iloc[0]['x'] if play_df['club'].isna().all() else play_df[play_df['club'] == 'football'].iloc[0]['x'])
        receiver_id, all_22_tracking_features = get_receiver_nflId(play, player_data, tracking_data)

        # print(f"\t{game_id},{play_id}")

        # receiver_id = None indicates pre-processing issue, so just skip
        try:
            # if receiver_id != None:
            receiver_x_at_pass = np.round(
                all_22_tracking_features[
                    (all_22_tracking_features['nflId'] == receiver_id) &
                    (all_22_tracking_features['event'].isin(['pass_forward', 'pass_shovel', 'autoevent_passforward']))
                ].iloc[0]['x']
            )

            # Only store plays in which the receiver is at or behind the LoS at the moment of the pass
            if receiver_x_at_pass - los <= 2:
                print(f"{game_id},{play_id} - LOS:{los}, RECEIVER X (at pass_forward): {receiver_x_at_pass}, result:{play['yardsGained' if 'yardsGained' in play.index else 'prePenaltyPlayResult']}")
                candidate_plays[(game_id, play_id)] = {'receiver_id': receiver_id, 
                                                       'tracking_data': all_22_tracking_features,
                                                       'los': los,
                                                       'receiver_x': receiver_x_at_pass,
                                                       'down': play['down'],
                                                       'yardsGained': play['yardsGained' if 'yardsGained' in play.index else 'prePenaltyPlayResult'],
                                                       'label': estimate_play_success(play)}
        except:
            skipped += 1
            continue
    
    print('TOTAL SKIPPED:', skipped)
    return candidate_plays




# Extract first name initial(s) and last name from playDescription
# Can handle cases such as "H.Henry", "Mi.Carter", and "A.St. Brown"
def extract_first_and_last_name(description):
    """
    Extracts first name initial(s) and last name from a play description.
    Returns a tuple: (initials, last_name), or (None, None) if not matched.
    """
    match = re.search(r'\bto\s+([A-Z][a-z]?)\.([A-Z][a-z]*(?:\.?\s?[A-Z][a-z]*)*)', description)
    if match:
        initials = match.group(1).strip()
        last_name = match.group(2).strip()
        return initials, last_name
    return None, None


def get_receiver_nflId(row, player_data: DataFrame, tracking_data: DataFrame):
    first_initial = row['receiver_first_initial']
    last_name = row['receiver_last_name']
    team = row['possessionTeam']
    game_id = row['gameId']
    play_id = row['playId']

    if pd.isnull(last_name):
        return None, None
    
    # Find frameId at the moment of the pass
    play_df = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
    pass_forward_frame_id = play_df[play_df['event'].isin(['pass_forward', 'pass_shovel', 'autoevent_passforward'])]['frameId'].min()

    # Get the spatiotemportal data of all 22 players at the moment of the pass
    all_22_tracking_features = play_df[play_df['frameId'] == pass_forward_frame_id]

    # Find player with a matching first initial and last name (case-insensitive)
    matches = player_data[
        (player_data['displayName'].str.startswith(first_initial)) &
        (player_data['displayName'].str.contains(fr'\b{re.escape(last_name)}\b', case=False, na=False))
    ]

    receiver_id = None
    if not matches.empty and len(matches) == 1: # 1 existing player with first initial and last name
        receiver_id = matches.iloc[0]['nflId']
    elif not matches.empty and len(matches) > 1: # more than 1 existing player with first initial and last name

        # Filter to only include players on offense who could receive the ball
        eligible_positions = ['WR', 'TE', 'RB', 'FB']
        skill_players_df = player_data[player_data['position'].isin(eligible_positions)]
        merged_df = play_df.merge(skill_players_df[['nflId', 'position']], on='nflId', how='left')
        possible_targets_df = merged_df[merged_df['position'].isin(eligible_positions)]
        possible_targets_df = possible_targets_df[(possible_targets_df['frameId'] == pass_forward_frame_id) & (possible_targets_df['club'] == team)]

        # Check for matches in this small subset, instead of all_players
        matches_in_possible_targets = matches[matches['nflId'].isin(possible_targets_df['nflId'])]['nflId'].values

        if len(matches_in_possible_targets) == 1:
            receiver_id = matches_in_possible_targets[0]

    return receiver_id, all_22_tracking_features




def estimate_play_success(play_data: DataFrame):
    down = play_data['down']

    yards_to_go = play_data['yardsToGo']
    yards_gained = play_data['yardsGained' if 'yardsGained' in play_data.index else 'playResult']
    yards_ratio = yards_gained / yards_to_go

    # Play succeeds if:
    #   40% of yardsToGo gained on 1st down
    #   60% of yardsToGo gained on 2nd down
    #   100% of yardsToGo gained on 3rd/4th down
    is_success = False
    if down == 1:
        is_success = True if yards_ratio >= 0.4 else False
    elif down == 2:
        is_success = True if yards_ratio >= 0.6 else False
    else:
        is_success = True if yards_ratio >= 1.0 else False
    
    return is_success



def create_input_tensor(play, play_data, player_data):
    print('play:', play)
    print('receiver:', play_data['receiver_id'])
    print(play_data)

    players_on_field = play_data['tracking_data']
    players_on_field = players_on_field[players_on_field['nflId'].notna()] # remove 'football' from tracking_data

    receiver_id = play_data['receiver_id']
    receiver = players_on_field[players_on_field['nflId'] == receiver_id].iloc[0]
    print('receiver:\n', receiver)

    # Remove the receiver from the tracking_data
    players_without_receiver = players_on_field[~(players_on_field['nflId'].isna() | (players_on_field['nflId'] == receiver_id))]
    # print(f'players_without_receiver ({len(players_without_receiver)}):\n', players_without_receiver)

    # Merge tracking_data with player positions in player_data
    merged_df = players_without_receiver.merge(player_data[['nflId', 'position']], on='nflId', how='left')

    #  Filter offensive and defensive players based on position
    off_players = merged_df[merged_df['position'].isin(constants.OFF_POSITIONS)].copy()
    def_players = merged_df[merged_df['position'].isin(constants.DEF_POSITIONS)].copy()

    print(f'off_players ({len(off_players)}):\n', off_players)
    print(f'def_players ({len(def_players)}):\n', def_players)

    # Get velocity of every player (including the receiver)
    player_velocities = {}
    for i,player in players_on_field.iterrows():
        player_nflId = player['nflId']
        player_speed = player['dis'] * 10 #player['s']
        player_dir = np.deg2rad(player['dir']) # TODO: Is this right?
        player_v_x = player_speed * np.cos(player_dir) # TODO: is it x=cos, y=sin, or vice versa?
        player_v_y = player_speed * np.sin(player_dir)

        player_velocities[player_nflId] = (player_v_x, player_v_y)

    # Calculate relative positions and speeds of every defender to receiver
    def_rel_pos = {}
    def_rel_vel = {}
    for i,defender in def_players.iterrows():
        rel_pos = (defender['x'] - receiver['x'], defender['y'] - receiver['y'])
        rel_vel = (player_velocities[defender['nflId']][0] - player_velocities[receiver_id][0], 
                     player_velocities[defender['nflId']][1] - player_velocities[receiver_id][1])

        def_rel_pos[defender['nflId']] = rel_pos
        def_rel_vel[defender['nflId']] = rel_vel

    
    # Calculate relative positions and speeds of every pair of offensive/defensive players (excluding receiver)
    off_def_pair_pos = {}
    off_def_pair_vel = {}
    for off_player, def_player in product(off_players.itertuples(index=False), def_players.itertuples(index=False)):
        diff_pos = (off_player.x - def_player.x, off_player.y - def_player.y)
        diff_vel = (player_velocities[off_player.nflId][0] - player_velocities[def_player.nflId][0], 
                     player_velocities[off_player.nflId][1] - player_velocities[def_player.nflId][1])
        
        off_def_pair_pos[(off_player.nflId, def_player.nflId)] = diff_pos
        off_def_pair_vel[(off_player.nflId, def_player.nflId)] = diff_vel
        

    # Construct tensor
    num_features = 5 * 2 # multiply by 2 because there is an (x,y) associated with each feature
    def_count = 11
    off_count = 10
    tensor = np.zeros((num_features, def_count, off_count))
    print('tensor:', tensor.shape)

    for i, def_player in enumerate(def_players.itertuples(index=False)):
        def_nflId = def_player.nflId

        # Get features relative to receiver
        def_v_x, def_v_y = player_velocities[def_nflId]
        rel_pos_x, rel_pos_y = def_rel_pos[def_nflId]
        rel_vel_x, rel_vel_y = def_rel_vel[def_nflId]

        for j, off_player in enumerate(off_players.itertuples(index=False)):
            off_nflId = off_player.nflId

            # Get features between defender and current offensive player
            off_def_rel_pos_x, off_def_rel_pos_y = off_def_pair_pos[(off_nflId, def_nflId)]
            off_def_rel_vel_x, off_def_rel_vel_y = off_def_pair_vel[(off_nflId, def_nflId)]

            # Fill tensor for current (def_player, off_player) pair
            # Channels 0-1: def velocity (v_x, v_y)
            tensor[0, i, j] = def_v_x
            tensor[1, i, j] = def_v_y

            # Channels 2-3: def position relative to receiver (x,y)
            tensor[2, i, j] = rel_pos_x
            tensor[3, i, j] = rel_pos_y

            # Channel 4-5: def velocity relative to receiver (v_x, v_y)
            tensor[4, i, j] = rel_vel_x
            tensor[5, i, j] = rel_vel_y

            # Channel 6-7: off - def position (x,y)
            tensor[6, i, j] = off_def_rel_pos_x
            tensor[7, i, j] = off_def_rel_pos_y

            # Channel 8-9: off - def velocity (v_x, v_y)
            tensor[8, i, j] = off_def_rel_vel_x
            tensor[9, i, j] = off_def_rel_vel_y




    print('==========================================')






    

def save_dict(dict, file_name):
    with open(f"{file_name}.pkl", 'wb') as f:
        pickle.dump(dict, f)


def get_dict(file_name):
    with open(f"{file_name}.pkl", 'rb') as f:
        dict = pickle.load(f)
    return dict



def scale_player_coordinates(player_x, player_y, x_scale=128/constants.FIELD_LENGTH, y_scale=64/constants.FIELD_WIDTH):
    x_scaled = player_x * x_scale
    y_scaled = player_y * y_scale
    return x_scaled, y_scaled


