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
import constants


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
    all_tracking_data_2021 = get_data.get_tracking_data(year=2021, week_start=1, week_end=1)
    all_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
    all_tracking_df_2021 = pd.concat(all_tracking_data_2021, ignore_index=True)
    all_play_data = get_data.get_play_data(year=2022)
    all_play_data_2021 = get_data.get_play_data(year=2021)
    all_player_data = get_data.get_player_data(year=2022)
    all_player_play_data = get_data.get_player_play_data(year=2022)

    # Filter to include only plays that contain a 'handoff' event
    handoff_plays = all_tracking_df[all_tracking_df['event'] == 'handoff'][['gameId', 'playId']]
    handoff_play_ids = set(zip(handoff_plays['gameId'], handoff_plays['playId']))
    run_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in handoff_play_ids, axis=1)]
    run_tracking_data = data_processing.filter_tracking_data(all_tracking_data, run_play_data)
    # run_tracking_data = data_processing.normalize_field_direction(run_tracking_data)

    # Filter to include only plays that contain a 'play_action' event
    play_action_plays = all_tracking_df[all_tracking_df['event'] == 'play_action'][['gameId', 'playId']]
    play_action_play_ids = set(zip(play_action_plays['gameId'], play_action_plays['playId']))
    play_action_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in play_action_play_ids, axis=1)]
    play_action_tracking_data = data_processing.filter_tracking_data(all_tracking_data, play_action_play_data)
    play_action_tracking_data = data_processing.normalize_field_direction(play_action_tracking_data)

    # Filter to include only RPO plays
    rpo_play_data = all_play_data[all_play_data['pff_runPassOption'] == 1]
    rpo_tracking_data = data_processing.filter_tracking_data(all_tracking_data, rpo_play_data)
    rpo_tracking_data = data_processing.normalize_field_direction(rpo_tracking_data)

    # Filter to include only TRADITIONAL dropback pass plays that have at least 2 seconds of timeInTackleBox
    # passing_play_data = all_play_data[(all_play_data['passResult'].notna()) & 
    #                                   (all_play_data['timeInTackleBox'] >= 2.0) &
    #                                   (all_play_data['dropbackType'] == 'TRADITIONAL')]
    passing_play_data = all_play_data[all_play_data['passResult'].notna()]
    passing_tracking_data = data_processing.filter_tracking_data(all_tracking_data, passing_play_data)
    passing_tracking_data = data_processing.normalize_field_direction(passing_tracking_data)

    # MAIN EXAMPLE: in (2021091206, 3353), 81 has a higher potential for yards, but QB throws to 28 instead
    # (2022091104, 3956): Goff could've passed it to 14 earlier and gotten a much larger gain
    # Good example (2022091104, 3204), (2022091100, 458), (2022091105, 4905), (2022091109, 743), (2022091112, 917)
    passing_play_data_2021 = all_play_data_2021[(all_play_data_2021['passResult'] == 'C')]# & (all_play_data_2021['playResult'] <= 3)]
    passing_tracking_data_2021 = data_processing.filter_tracking_data(all_tracking_data_2021, passing_play_data_2021)
    passing_tracking_data_2021 = data_processing.normalize_field_direction(passing_tracking_data_2021)

    # Filter to include only pass plays that were thrown within 1 yards of the LoS
    passes_behind_los_play_data = passing_play_data[(passing_play_data['passResult'] == 'C') & 
                                                    (passing_play_data['passLength'] <= 2) &
                                                    (passing_play_data['passTippedAtLine'] == False) &
                                                    (passing_play_data['playNullifiedByPenalty'] == 'N')]# &
                                                    # (passing_play_data['targetY'] >= constants.SIDELINE_TO_HASH / 2) &
                                                    # (passing_play_data['targetY'] < constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH / 2)]
    passes_behind_los_tracking_data = data_processing.filter_tracking_data(all_tracking_data, passes_behind_los_play_data)
    passes_behind_los_tracking_data = data_processing.normalize_field_direction(passes_behind_los_tracking_data)
    

    # passes_behind_los_play_2021_data = all_tracking_df[all_tracking_df_2021['event'] == 'play_action'][['gameId', 'playId']]
    # play_action_play_ids = set(zip(passes_behind_los_play_2021_data['gameId'], passes_behind_los_play_2021_data['playId']))
    # play_action_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in play_action_play_ids, axis=1)]



    print('# of handoff plays:\t', len(run_play_data))
    print('# of play-action plays:\t', len(all_play_data[all_play_data['playAction'] == True]))
    print('# of RPO plays:', len(rpo_play_data))
    print('# of passing plays:', len(passing_play_data))
    print('# of passing plays behind LoS:', len(passes_behind_los_play_data))
    print('#\t Average EPA on passes behind LoS:' ,passes_behind_los_play_data['expectedPointsAdded'].mean())

    median_yardsGained_yardsToGo_ratio = (passes_behind_los_play_data['yardsGained'] / passes_behind_los_play_data['yardsToGo']).median()
    plays_above_yardsGained_yardsToGo_ratio = passes_behind_los_play_data[(passes_behind_los_play_data['yardsGained']/passes_behind_los_play_data['yardsToGo']) >= median_yardsGained_yardsToGo_ratio]
    print('#\t Mean yardsGained/yardsToGo ratio on passes behind LoS:', median_yardsGained_yardsToGo_ratio)
    print('#\t Percent of behind LoS passes >= median_yardsGained_yardsToGo_ratio:', len(plays_above_yardsGained_yardsToGo_ratio) / len(passes_behind_los_play_data))
    print('#\t Max yardsGained on passes behind LoS:', passes_behind_los_play_data['yardsGained'].max())
    print('# of 2021 passing plays:', len(all_play_data_2021))
    # print('# of 2021 completed passing plays:', len(passing_play_data_2021))

    # print('ALL RUSH EVENTS:', run_tracking_data[0]['event'].value_counts())
    # print('ALL RUSH O FORMATIONS:', run_play_data['offenseFormation'].value_counts())
    # print('ALL RUSH D FORMATIONS:', run_play_data['pff_passCoverage'].value_counts())
    # print('ALL RUSH rushLocationType:', run_play_data['rushLocationType'].value_counts())
    # print('ALL RUSH pff_runConceptPrimary:', run_play_data['pff_runConceptPrimary'].value_counts())
    # print('ALL RUSH pff_runConceptSecondary:', run_play_data['pff_runConceptSecondary'].value_counts())
    # print('ALL PA EVENTS:', play_action_tracking_data[0]['event'].value_counts())
    # print('ALL PA O FORMATIONS:', play_action_play_data['offenseFormation'].value_counts())
    # print('ALL PA D FORMATIONS:', play_action_play_data['pff_passCoverage'].value_counts())
    # print('ALL RPO EVENTS:', rpo_tracking_data[0]['event'].value_counts())
    # print('ALL RPO O FORMATIONS:', rpo_play_data['offenseFormation'].value_counts())
    # print('ALL RPO D FORMATIONS:', rpo_play_data['pff_passCoverage'].value_counts())
    # print('ALL PASSING DROPBACK TYPES:\n', passing_play_data['dropbackType'].value_counts())
    # print('AVG TIME TO THROW:', passing_play_data['timeToThrow'].mean())
    # print('AVG timeToPressureAsPassRusher:', all_player_play_data['timeToPressureAsPassRusher'].mean())
    # print('AVG getOffTimeAsPassRusher:', all_player_play_data['getOffTimeAsPassRusher'].mean())

    sample_num = 1

    # test_pa_plays = random.sample(range(len(play_action_play_data)), sample_num)
    # play_action_play_data = play_action_play_data[play_action_play_data['gameId'] <= 2022091200] # Week 1 
    # play_action_frames_dict = data_processing.get_relevant_frames(play_action_play_data.iloc[test_pa_plays], play_action_tracking_data, start_events=['line_set'], end_events=['END']) # end_events=['play_action']

    # test_run_plays = random.sample(range(len(run_play_data)), sample_num)
    # run_play_data = run_play_data[run_play_data['gameId'] <= 2022091200] # Week 1 only
    # run_frames_dict = data_processing.get_relevant_frames(run_play_data.iloc[test_run_plays], run_tracking_data, start_events=['line_set'], end_events=['END']) # end_events=['handoff']

    # rpo_play_data = rpo_play_data[rpo_play_data['gameId'] <= 2022091200] # Week 1 only
    # test_rpo_plays = random.sample(range(len(rpo_play_data)), sample_num)
    # rpo_frames_dict = data_processing.get_relevant_frames(rpo_play_data.iloc[test_rpo_plays], rpo_tracking_data, start_events=['line_set'], end_events=['END'])

    passes_behind_los_play_data = passes_behind_los_play_data[passes_behind_los_play_data['gameId'] <= 2022091200] # Week 1 only
    test_passes_behind_los_plays = random.sample(range(len(passes_behind_los_play_data)), sample_num)
    passes_behind_los_frames_dict = data_processing.get_relevant_frames(passes_behind_los_play_data.iloc[test_passes_behind_los_plays], passes_behind_los_tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_ARRIVED])

    passing_play_data_2021 = passing_play_data_2021[(passing_play_data_2021['gameId'] <= 2021091300)] # Week 1 only
    test_pass_plays_2021 = random.sample(range(len(passing_play_data_2021)), sample_num)
    passes_2021_dict = data_processing.get_relevant_frames(passing_play_data_2021.iloc[test_pass_plays_2021], all_tracking_data_2021, start_events=[constants.BALL_SNAP], end_events=[constants.END])





    # passing_play_data = passing_play_data[passing_play_data['gameId'] <= 2022091200] # Week 1 only

    # random_tests = random.sample(range(len(passing_play_data)), sample_num)
    # random_test_passing_plays = passing_play_data.iloc[random_tests]
    # use_cases = [(2022091101, 1879), (2022091109, 1041), (2022091108, 614), (2022091101, 1492), (2022091101, 1166),
    #             (2022091105, 2817), (2022091108, 2799), (2022091107, 1642), (2022091106, 3050), (2022091111, 336),
    #             (2022091105, 2817), (2022091104, 2952)]

    # # Filter all passing_play_data to use_cases
    # test_case_set = set(use_cases)
    # use_case_plays = passing_play_data[
    #     passing_play_data.apply(lambda row: (row['gameId'], row['playId']) in test_case_set, axis=1)
    # ]

    # plays_to_process = pd.concat([random_test_passing_plays, use_case_plays], ignore_index=True)
    
    # # Create GIFs of 1 randomly selected play
    # play_to_gif = random_test_passing_plays
    # passing_frames_dict = data_processing.get_relevant_frames(play_to_gif, passing_tracking_data, start_events=['line_set'], end_events=['END'])



    # Calculate label for each play
    passes_behind_los_success_labels = data_processing.estimate_play_success(passes_behind_los_play_data, 2022)


    passes_behind_los_tracking_data = pd.concat(passes_behind_los_tracking_data, ignore_index=True)
    for i,play in passes_behind_los_play_data[5:10].iterrows():
        game_id = play['gameId']
        play_id = play['playId']
        possession_team = play['possessionTeam']

        play_df = passes_behind_los_tracking_data[(passes_behind_los_tracking_data['gameId'] == game_id) & (passes_behind_los_tracking_data['playId'] == play_id)]

        # Get the frameId of the 'pass_forward' event
        pass_forward_frame_id = play_df[play_df['event'] == 'pass_forward']['frameId'].min()
        target_frame_id = pass_forward_frame_id + 3

        # Get ball trajectory data as the pass is thrown (use a few frames ahead to calculate direction)
        ball_frame0 = play_df[(play_df['frameId'] == pass_forward_frame_id) & (play_df['club'] == 'football')].iloc[0]
        ball_frame1 = play_df[(play_df['frameId'] == target_frame_id) & (play_df['club'] == 'football')].iloc[0]
        ball_x0, ball_y0 = ball_frame0['x'], ball_frame0['y']
        ball_x1, ball_y1 = ball_frame1['x'], ball_frame1['y']
        ball_direction_vector = np.array([ball_x1 - ball_x0, ball_y1 - ball_y0])
        ball_dx, ball_dy = ball_direction_vector / (np.linalg.norm(ball_direction_vector) + 1e-6)
        theta_rad = np.arctan2(ball_dx, ball_dy) # convert to radians
        ball_dir_angle = (np.rad2deg(theta_rad) + 360) % 360 # convert to degrees 0-360

        ball_data = {
            'x': ball_frame0['x'],
            'y': ball_frame0['y'],
            'dir': ball_dir_angle
        }

        offense_players = play_df[(play_df['frameId'] == target_frame_id) & (play_df['club'] == possession_team)]

        print(f"{game_id},{play_id} - ball data: {ball_data}\noffensive players:\n{offense_players}")
        



    # Create GIFs for passing plays
    # for play,play_frames in passing_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = passing_play_data[(passing_play_data['gameId'] == game_id) & (passing_play_data['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_passing_norm', loop=False, zoom=False)

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

    # Create GIFs for passes plays behind the LoS
    # for play,play_frames in passes_behind_los_frames_dict.items():
    #     game_id, play_id = play
    #     play_data = passes_behind_los_play_data[(passes_behind_los_play_data['gameId'] == game_id) & (passes_behind_los_play_data['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm', loop=False, zoom=False)

    # Create GIFs for 2021 passes
    # for play,play_frames in passes_2021_dict.items():
    #     game_id, play_id = play
    #     play_data = passing_play_data_2021[(passing_play_data_2021['gameId'] == game_id) & (passing_play_data_2021['playId'] == play_id)].iloc[0]
    #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_2021_norm_short', loop=False, zoom=False)








if __name__ == "__main__":
    main()