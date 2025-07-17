import numpy as np
# import get_data
# from passing_down import PassingDown
# import random_tree
# import fnn
import time
import pandas as pd
import os
# from fnn import FNN
import data_processing
import get_data
import torch
import random
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import visualization
import random
import math


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)


def main():
    print('running main')
    # set_seed()

    is_testing = True

    # Obtain all play and tracking data
    all_tracking_data = get_data.get_tracking_data(year=2022, week_start=1, week_end=1)
    all_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
    all_play_data = get_data.get_play_data(year=2022)
    all_player_data = get_data.get_player_data(year=2022)
    all_player_play_data = get_data.get_player_play_data(year=2022)

    # Filter to include only plays that contain a 'handoff' event
    # handoff_plays = all_tracking_df[all_tracking_df['event'] == 'handoff'][['gameId', 'playId']]
    # handoff_play_ids = set(zip(handoff_plays['gameId'], handoff_plays['playId']))
    # run_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in handoff_play_ids, axis=1)]
    # run_tracking_data = data_processing.filter_tracking_data(all_tracking_data, run_play_data)
    # run_tracking_data = data_processing.normalize_field_direction(run_tracking_data)

    # Filter to include only plays that contain a 'play_action' event
    # play_action_plays = all_tracking_df[all_tracking_df['event'] == 'play_action'][['gameId', 'playId']]
    # play_action_play_ids = set(zip(play_action_plays['gameId'], play_action_plays['playId']))
    # play_action_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in play_action_play_ids, axis=1)]
    # play_action_tracking_data = data_processing.filter_tracking_data(all_tracking_data, play_action_play_data)
    # play_action_tracking_data = data_processing.normalize_field_direction(play_action_tracking_data)

    # Filter to include only RPO plays
    # rpo_play_data = all_play_data[all_play_data['pff_runPassOption'] == 1]
    # rpo_tracking_data = data_processing.filter_tracking_data(all_tracking_data, rpo_play_data)
    # rpo_tracking_data = data_processing.normalize_field_direction(rpo_tracking_data)

    # Filter to include only TRADITIONAL dropback pass plays that have at least 2 seconds of timeInTackleBox
    passing_play_data = all_play_data[(all_play_data['passResult'].notna()) & 
                                      (all_play_data['timeInTackleBox'] >= 2.0) &
                                      (all_play_data['dropbackType'] == 'TRADITIONAL')]
    passing_tracking_data = data_processing.filter_tracking_data(all_tracking_data, passing_play_data)
    passing_tracking_data = data_processing.normalize_field_direction(passing_tracking_data)


    # print('# of handoff plays:\t', len(run_play_data))
    # print('# of play-action plays:\t', len(play_action_play_data))
    # print('# of RPO plays:', len(rpo_play_data))
    print('# of passing plays:', len(passing_play_data))

    # print('ALL RUSH EVENTS:', run_tracking_data[0]['event'].value_counts())
    # print('ALL RUSH O FORMATIONS:', run_play_data['offenseFormation'].value_counts())
    # print('ALL RUSH D FORMATIONS:', run_play_data['pff_passCoverage'].value_counts())
    # print('ALL PA EVENTS:', play_action_tracking_data[0]['event'].value_counts())
    # print('ALL PA O FORMATIONS:', play_action_play_data['offenseFormation'].value_counts())
    # print('ALL PA D FORMATIONS:', play_action_play_data['pff_passCoverage'].value_counts())
    # print('ALL RPO EVENTS:', rpo_tracking_data[0]['event'].value_counts())
    # print('ALL RPO O FORMATIONS:', rpo_play_data['offenseFormation'].value_counts())
    # print('ALL RPO D FORMATIONS:', rpo_play_data['pff_passCoverage'].value_counts())
    # print('ALL PASSING DROPBACK TYPES:\n', passing_play_data['dropbackType'].value_counts())
    print('AVG TIME TO THROW:', passing_play_data['timeToThrow'].mean())

    sample_num = 1

    if is_testing:

        # test_pa_plays = random.sample(range(len(play_action_play_data)), sample_num)
        # play_action_play_data = play_action_play_data[play_action_play_data['gameId'] <= 2022091200] # Week 1 
        # # print(play_action_play_data.iloc[test_pa_play])
        # play_action_frames_dict = data_processing.get_relevant_frames(play_action_play_data.iloc[test_pa_plays], play_action_tracking_data, start_events=['line_set'], end_events=['END']) # end_events=['play_action']

        # test_run_plays = random.sample(range(len(run_play_data)), sample_num)
        # run_play_data = run_play_data[run_play_data['gameId'] <= 2022091200] # Week 1 only
        # # print(run_play_data.iloc[test_run_play])
        # run_frames_dict = data_processing.get_relevant_frames(run_play_data.iloc[test_run_plays], run_tracking_data, start_events=['line_set'], end_events=['END']) # end_events=['handoff']

        # rpo_play_data = rpo_play_data[rpo_play_data['gameId'] <= 2022091200] # Week 1 only
        # test_rpo_plays = random.sample(range(len(rpo_play_data)), sample_num)
        # # print(rpo_play_data.iloc[test_rpo_play])
        # rpo_frames_dict = data_processing.get_relevant_frames(rpo_play_data.iloc[test_rpo_plays], rpo_tracking_data, start_events=['line_set'], end_events=['END'])


        passing_play_data = passing_play_data[passing_play_data['gameId'] <= 2022091200][:10] # Week 1 only
        test_passing_plays = random.sample(range(len(passing_play_data)), sample_num)
        passing_frames_dict = data_processing.get_relevant_frames(passing_play_data.iloc[test_passing_plays], passing_tracking_data, start_events=['line_set'], end_events=['END'])

        defense_rush_positions = ['CB', 'OLB', 'DE', 'DT', 'ILB', 'FS', 'SS', 'NT', 'MLB', 'DB', 'LB']
        all_def_players = all_player_data[all_player_data['position'].isin(defense_rush_positions)]['nflId'].unique()
        for i,passing_play in passing_play_data.iterrows():
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
            pc_heat_val = len(rushing_defenders_pressure_caused) / len(rushing_defenders)

            avg_time_to_pressure = np.round(np.sum(total_times_to_pressure) / len(rushing_defenders), 2)
            thresh = 2 # to represent 'quick' throw
            # ttp_heat_val = np.round(1 - (avg_time_to_pressure / max_time_to_pass), 2)
            ttp_heat_val = np.round(np.clip(1 / (1 + np.exp(avg_time_to_pressure - thresh)), 0, 1), 2)


            # The max amount of time it would take to cross LoS as rusher
            # TODO: try and estimate this number
            max_getoff_time = 2
            avg_time_to_getoff = np.round(np.median(total_times_to_getoff) / len(rushing_defenders), 2) #median instead of mean for robustness to outliers (e.g., one slow rusher)
            ttg_heat_val = np.round(1 - (avg_time_to_getoff / max_getoff_time), 2)

            # Time to throw
            # Did QB have to bail from the pocket?
            # ttt_heat_val = np.round(1 - (passing_play['timeInTackleBox'] / time_to_pass_result), 2)
            ttt_heat_val = np.round(1 - np.clip(passing_play['timeInTackleBox'] / time_to_pass_result, 0, 1) ** 0.5, 2) # square root emphasizes short times


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

            penalties = 0
            if passing_play['unblockedPressure']:
                penalities += 0.2


            # EXTREMELY CLEAN POCKET: 2022091113, 630
            # IMMEDIATE PRESSURE FROM 1 DEFENDER: 2022091109, 1041
            print('==========================================================================================')
            print(f'# OF RUSHERS ON ({game_id}, {play_id}): {len(rushing_defenders)} ({len(rushing_defenders_pressure_caused)} caused pressure) ({pc_heat_val} heat val)')
            print(f'Time to pass result: {time_to_pass_result}')
            print(f'Avg time to pressure for rushing defenders: {avg_time_to_pressure} ({ttp_heat_val} heat val)')
            print(f'Avg time to getoff for rushing defenders: {avg_time_to_getoff} ({ttg_heat_val} heat val)')
            print(f"QB time in tackle box: {passing_play['timeInTackleBox']} ({ttt_heat_val} heat val)")
            # print(f"Outcome: {'SACK' if not math.isnan(passing_play['timeToSack']) else 'NO SACK'} ({outcome_heat_val})")
            print(f'TOTAL HEAT VAL: {np.round(pc_heat_val + ttp_heat_val + ttg_heat_val + ttt_heat_val + penalties, 2)}')



    # Create GIFs for passing plays
    for play,play_frames in passing_frames_dict.items():
        game_id, play_id = play
        play_data = passing_play_data[(passing_play_data['gameId'] == game_id) & (passing_play_data['playId'] == play_id)].iloc[0]
        visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_passing_norm', loop=False, zoom=False)

    # Create GIFs for RPO plays
    # for play,play_frames in rpo_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = rpo_play_data[(rpo_play_data['gameId'] == game_id) & (rpo_play_data['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_rpo_norm', loop=False, zoom=False)
    
    # Create GIFs for play-action plays
    # for play,play_frames in play_action_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = play_action_play_data[(play_action_play_data['gameId'] == game_id) & (play_action_play_data['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_pa_norm', loop=False, zoom=False)

    # Create GIFs for run plays
    # for play,play_frames in run_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = run_play_data[(run_play_data['gameId'] == game_id) & (run_play_data['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_run_norm', loop=False, zoom=False)





















    # passing_play_data, passing_tracking_data = get_data.get_passing_plays(year=2022, week_start=1, week_end=1 if is_testing else 9)
    # player_data = get_data.get_player_data(year=2022)

    # print('passing play data:', len(passing_play_data))
    # print('tracking data week #1:', len(passing_tracking_data))

    # if is_testing:
    #     # 900 for left dir
    #     test_play = 216#800 #705, 724, 105, 469, 658

    #     passing_play_data = passing_play_data[passing_play_data['gameId'] <= 2022091200] # Week 1 only # 2022090800, 2022091200
    #     print(passing_play_data.iloc[test_play])

    #     passing_frames_dict = data_processing.get_pocket_frames(passing_play_data.iloc[[test_play]], passing_tracking_data) #passing_play_data.iloc[[0]]
    #     print(passing_frames_dict.keys())

    # defense_rush_positions = ['CB', 'OLB', 'DE', 'DT', 'ILB', 'FS', 'SS', 'NT', 'MLB', 'DB', 'LB']
    # all_def_players = player_data[player_data['position'].isin(defense_rush_positions)]['nflId'].unique()
    # all_qbs = player_data[player_data['position'] == 'QB']['nflId'].unique()

    # for play,play_frames in passing_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = passing_play_data[(passing_play_data['gameId'] == game_id) & (passing_play_data['playId'] == play_id)].iloc[0]

    #     visualization.plot_frame(play_frames, play_data, f'{game_id}_{play_id}_norm', zoom=True)

    #     # Get ball location
    #     ball_x, ball_y = play_frames[(play_frames['displayName'] == 'football') & (play_frames['event'] == 'ball_snap')].iloc[0][['x', 'y']] #play_frames.iloc[0][['x', 'y']]
    #     print('ball_coords:', ball_x, ball_y)

    #     # Get QB location
    #     qb = play_frames[play_frames['nflId'].isin(all_qbs)].iloc[0]
    #     qb_x, qb_y = qb[['x', 'y']]
    #     qb_display = qb['displayName']
    #     print(qb_display, qb_x, qb_y)

    #     print('PLAY:\n', play_data)

        # Normalize the ball and all players to center of field
        # centered_play_frames = data_processing.normalize_to_center(play_frames, (ball_x, ball_y))
        # data_processing.plot_frame(centered_play_frames, play_data, f'{game_id}_{play_id}_norm_centered')

        # data_processing.detect_rushers(all_def_players, play_frames, (ball_x, ball_y), (qb_x, qb_y))



if __name__ == "__main__":
    main()