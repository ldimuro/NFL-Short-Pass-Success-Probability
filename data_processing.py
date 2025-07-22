import pandas as pd
from pandas import DataFrame
import numpy as np
import constants
import math
import re


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

        # print(f'searching for {game_id} - {play_id}')

        # Look through all weeks of tracking data for specific play
        tracking_play = None
        for i,week_df in enumerate(tracking_data):
            match = week_df[(week_df['gameId'] == game_id) & (week_df['playId'] == play_id)]
            
            if not match.empty:
                tracking_play = match.sort_values('frameId')
                # print('FOUND in week', i+1)
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
                print(f"🚨Warning: Missing event in play ({game_id},{play_id})")

            # Add all relevant tracking frames to plays dict
            play_tracking_dict[(game_id, play_id)] = tracking_play
            # print(f'processed tracking frames for {game_id} - {play_id}')
        else:
            print(f'🚨could not find {game_id} - {play_id}')

    return play_tracking_dict



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
    pass_forward_frame_id = play_df[play_df['event'].isin(['pass_forward', 'autoevent_passforward'])]['frameId'].min()

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




# Input: play data, Output: (gameId, playId), Success=1/Failure=0
def estimate_play_success(play_data: DataFrame, year):
    play_success_labels = {}
    for i,play in play_data.iterrows():
        game_id = play['gameId']
        play_id = play['playId']
        down = play['down']

        yards_to_go = play['yardsToGo']
        yards_gained = play['playResult' if year == 2021 else 'yardsGained']
        yards_ratio = yards_gained / yards_to_go

        # Play succeeds if:
        #   40% of yardsToGo gained on 1st down
        #   60% of yardsToGo gained on 2nd down
        #   100% of yardsToGo gained on 3rd/4th down
        if down == 1:
            is_success = True if yards_ratio >= 0.4 else False
        elif down == 2:
            is_success = True if yards_ratio >= 0.6 else False
        else:
            is_success = True if yards_ratio >= 1.0 else False
        
        play_success_labels[(game_id, play_id)] = is_success

    return play_success_labels



    


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


# Calculate how "hot" a pocket is
def calculate_pocket_heat(plays_to_process, all_player_data, all_player_play_data, all_tracking_df):

    # EXTREMELY CLEAN POCKET: (2022091113, 630), (2022091101, 1879), (2022091104, 2952)
            # IMMEDIATE PRESSURE FROM 1 DEFENDER: (2022091109, 1041)
            # TOTAL COLLAPSE: (2022091108, 614)
            # Scored very low: (2022091101, 1492)
            # Starts off completely clean, but then has an unblocked rusher (not labelled): (2022091101, 1166)
            # Moderate pressure?: (2022091105, 2817)
            # Scored average, but is very good protection: (2022091108, 2799)
            # Avg score but feels like it should be higher (2022091107, 1642)
            # Dont know why it statistically has a high score: (2022091106, 3050)
            # Very high score: (2022091111, 336)
            # Moderate pressure: (2022091105, 2817)
    defense_rush_positions = ['CB', 'OLB', 'DE', 'DT', 'ILB', 'FS', 'SS', 'NT', 'MLB', 'DB', 'LB']
    all_def_players = all_player_data[all_player_data['position'].isin(defense_rush_positions)]['nflId'].unique()
    for i,passing_play in plays_to_process.iterrows():
        game_id = passing_play['gameId']
        play_id = passing_play['playId']

        play_tracking = all_tracking_df[(all_tracking_df['gameId'] == game_id) & (all_tracking_df['playId'] == play_id)]
        players_in_play = play_tracking['nflId'].dropna().unique()
        def_players_in_play = np.intersect1d(players_in_play, all_def_players)

        # Calculate the time it took from ball-snap to either timeToThrow or timeToSack
        time_to_pass_result = passing_play['timeToThrow'] if not math.isnan(passing_play['timeToThrow']) else passing_play['timeToSack']

        # Extract "pressure" features
        rushing_defenders = []
        rushing_defenders_pressure_caused = []
        total_times_to_pressure = []
        total_times_to_getoff = []

        for def_id in def_players_in_play:
            player= all_player_data[all_player_data['nflId'] == def_id].iloc[0]

            def_play_data = all_player_play_data[
                (all_player_play_data['gameId'] == game_id) &
                (all_player_play_data['playId'] == play_id) &
                (all_player_play_data['nflId'] == def_id)
            ].iloc[0]

            caused_pressure = def_play_data['causedPressure']
            time_to_pressure_as_rusher = def_play_data['timeToPressureAsPassRusher']
            get_off_as_rusher = def_play_data['getOffTimeAsPassRusher']

            # Count number of rushers (who have a non-NaN value for 'getOffTimeAsPassRusher')
            if not math.isnan(get_off_as_rusher):
                rushing_defenders.append(def_id)
                total_times_to_getoff.append(get_off_as_rusher)

                if caused_pressure:
                    rushing_defenders_pressure_caused.append(def_id)
                    total_times_to_pressure.append(time_to_pressure_as_rusher)

            # print(f"{player['displayName']} play data:")
            # print(f'\tcaused_pressure: {caused_pressure}')
            # print(f'\ttime_to_pressure_as_rusher: {time_to_pressure_as_rusher}')
            # print(f'\tget_off_as_rusher: {get_off_as_rusher}')


        # If there was no pressure on the play, assign arbitrary large value
        if len(total_times_to_pressure) == 0:
            total_times_to_pressure = [5]


        # Ratio of rushers to rushers that caused pressure
        # pc_heat_val = len(rushing_defenders_pressure_caused) / len(rushing_defenders) * 10
        pc_heat_val = 10 * len(rushing_defenders_pressure_caused)

        avg_time_to_pressure = np.round(np.sum(total_times_to_pressure) / len(rushing_defenders), 2)
        thresh = 2 # to represent 'quick' throw
        # ttp_heat_val = np.round(1 - (avg_time_to_pressure / max_time_to_pass), 2)
        ttp_heat_val = 0 #np.round(np.clip(1 / (1 + np.exp(avg_time_to_pressure - thresh)), 0, 1), 2)


        # The max amount of time it would take to cross LoS as rusher
        # TODO: try and estimate this number
        max_getoff_time = 2
        avg_time_to_getoff = np.round(np.median(total_times_to_getoff) / len(rushing_defenders), 2) #median instead of mean for robustness to outliers (e.g., one slow rusher)
        ttg_heat_val = np.round(1 - (avg_time_to_getoff / max_getoff_time), 2) * 10

        # Time to throw
        # Did QB have to bail from the pocket?
        # ttt_heat_val = np.round(1 - (passing_play['timeInTackleBox'] / time_to_pass_result), 2)
        ttt_heat_val = np.round(1 - np.clip(passing_play['timeInTackleBox'] / time_to_pass_result, 0, 1) ** 0.5, 2) * 10 # square root emphasizes short times

        

        # Outcome
        # Max time limit for a clean pocket
        # TODO: try and estimate this number
        # max_time_to_pass = 5
        # # outcome_heat_val = np.round(1 - (passing_play['timeToThrow'] if not math.isnan(passing_play['timeToThrow']) else passing_play['timeToSack'] / max_time_to_pass), 2)
        # # outcome_heat_val = np.clip(outcome_heat_val, 0, 1)
        # if not math.isnan(passing_play['timeToSack']):
        #     outcome_heat_val = np.round(passing_play['timeToSack'] / max_time_to_pass, 2)
        #     outcome_heat_val += 0.2 # boosted for sacks
        # else:
        #     outcome_heat_val = np.round(passing_play['timeToThrow'] / max_time_to_pass, 2)
        #     outcome_heat_val *= 0.5 # half-weight for non-sacks, as quick throws indicate indirect heat
        # outcome_heat_val = np.clip(outcome_heat_val, 0, 1)

        # TODO: Add penalty for # of defenders within 1-2 yards at the time of the league average timeToThrow (2.7)

        penalties = 0
        if passing_play['unblockedPressure']:
            print('Unblocked pressure! (10 heat val)')
            penalties += 10

        
        # TODO: Add bonus for time over league average (2.7) while remaining in the Tackle Box

        # TODO: Alternate idea: Make the "pocket_heat" value per frame, so the model can see the pocket_heat change over the course of the play
        #       Use the distance of the QB to every rusher at every frame to assign score

        


        
        print('==========================================================================================')
        print(f'# OF RUSHERS ON ({game_id}, {play_id}): {len(rushing_defenders)} ({len(rushing_defenders_pressure_caused)} caused pressure) ({pc_heat_val} heat val)')
        print(f'Time to pass result: {time_to_pass_result}')
        print(f'Avg time to pressure for rushing defenders: {avg_time_to_pressure} ({ttp_heat_val} heat val)')
        print(f'Avg time to getoff for rushing defenders: {avg_time_to_getoff} ({ttg_heat_val} heat val)')
        print(f"QB time in tackle box: {passing_play['timeInTackleBox']} ({ttt_heat_val} heat val)")
        print(f"Penalties ({penalties} heat val)")
        # print(f"Outcome: {'SACK' if not math.isnan(passing_play['timeToSack']) else 'NO SACK'} ({outcome_heat_val})")
        print(f'TOTAL HEAT VAL: {np.round(pc_heat_val + ttp_heat_val + ttg_heat_val + ttt_heat_val + penalties, 2)}')


def scale_player_coordinates(player_x, player_y, x_scale=128/constants.FIELD_LENGTH, y_scale=64/constants.FIELD_WIDTH):
    x_scaled = player_x * x_scale
    y_scaled = player_y * y_scale
    return x_scaled, y_scaled


