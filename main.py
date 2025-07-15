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


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)


def main():
    print('running main')
    set_seed()

    is_testing = True

    # Obtain all play and tracking data
    all_tracking_data = get_data.get_tracking_data(year=2022, week_start=1, week_end=1)
    all_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
    all_play_data = get_data.get_play_data(year=2022)

    # Filter to include only plays that contain a 'handoff' event
    handoff_plays = all_tracking_df[all_tracking_df['event'] == 'handoff'][['gameId', 'playId']]
    handoff_play_ids = set(zip(handoff_plays['gameId'], handoff_plays['playId']))
    run_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in handoff_play_ids, axis=1)]
    run_tracking_data = data_processing.filter_tracking_data(all_tracking_data, run_play_data)
    run_tracking_data = data_processing.normalize_field_direction(run_tracking_data)

    # Filter to include only plays that contain a 'play_action' event
    play_action_plays = all_tracking_df[all_tracking_df['event'] == 'play_action'][['gameId', 'playId']]
    play_action_play_ids = set(zip(play_action_plays['gameId'], play_action_plays['playId']))
    play_action_play_data = all_play_data[all_play_data.apply(lambda row: (row['gameId'], row['playId']) in play_action_play_ids, axis=1)]
    play_action_tracking_data = data_processing.filter_tracking_data(all_tracking_data, play_action_play_data)
    play_action_tracking_data = data_processing.normalize_field_direction(play_action_tracking_data)


    # Remove all plays less than 10 frames
    # Make sure there is no overlap between play-action and run plays
    # Remove plays that involve a play-action into a handoff

    print('# of handoff plays:\t', len(run_play_data))
    print('# of play-action plays:\t', len(play_action_play_data))

    print('ALL RUSH EVENTS:', run_tracking_data[0]['event'].value_counts())
    print('ALL PA EVENTS:', play_action_tracking_data[0]['event'].value_counts())

    sample_num = 3


    if is_testing:
        test_pa_plays = random.sample(range(len(play_action_play_data)), sample_num)

        play_action_play_data = play_action_play_data[play_action_play_data['gameId'] <= 2022091200] # Week 1 only # 2022090800, 2022091200
        # print(play_action_play_data.iloc[test_pa_play])
        play_action_frames_dict = data_processing.get_relevant_frames(play_action_play_data.iloc[test_pa_plays], play_action_tracking_data, start_events=['line_set'], end_events=['play_action']) #passing_play_data.iloc[[0]]

        test_run_plays = random.sample(range(len(run_play_data)), sample_num)
        run_play_data = run_play_data[run_play_data['gameId'] <= 2022091200] # Week 1 only # 2022090800, 2022091200
        # print(run_play_data.iloc[test_run_play])
        run_frames_dict = data_processing.get_relevant_frames(run_play_data.iloc[test_run_plays], run_tracking_data, start_events=['line_set'], end_events=['handoff']) #passing_play_data.iloc[[0]]


    
    for play,play_frames in play_action_frames_dict.items():
        game_id, play_id = play
        play_data = play_action_play_data[(play_action_play_data['gameId'] == game_id) & (play_action_play_data['playId'] == play_id)].iloc[0]
        # visualization.plot_frame(play_frames, play_data, f'{game_id}_{play_id}_norm', zoom=True)
        visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_pa_norm', loop=False, zoom=False)

    for play,play_frames in run_frames_dict.items():
        game_id, play_id = play
        play_data = run_play_data[(run_play_data['gameId'] == game_id) & (run_play_data['playId'] == play_id)].iloc[0]
        # visualization.plot_frame(play_frames, play_data, f'{game_id}_{play_id}_norm', zoom=True)
        visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_run_norm', loop=False, zoom=False)





















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