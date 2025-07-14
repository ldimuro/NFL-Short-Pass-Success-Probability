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

    passing_play_data, passing_tracking_data = get_data.get_passing_plays(year=2022, week_start=1, week_end=1 if is_testing else 9)
    player_data = get_data.get_player_data(year=2022)

    print('passing play data:', len(passing_play_data))
    print('tracking data week #1:', len(passing_tracking_data))

    if is_testing:
        # 900 for left dir
        test_play = 216#800 #705, 724, 105, 469, 658

        passing_play_data = passing_play_data[passing_play_data['gameId'] <= 2022091200] # Week 1 only # 2022090800, 2022091200
        print(passing_play_data.iloc[test_play])

        passing_frames_dict = data_processing.get_pocket_frames(passing_play_data.iloc[[test_play]], passing_tracking_data) #passing_play_data.iloc[[0]]
        print(passing_frames_dict.keys())


    defense_rush_positions = ['CB', 'OLB', 'DE', 'DT', 'ILB', 'FS', 'SS', 'NT', 'MLB', 'DB', 'LB']
    all_def_players = player_data[player_data['position'].isin(defense_rush_positions)]['nflId'].unique()
    all_qbs = player_data[player_data['position'] == 'QB']['nflId'].unique()

    for play,play_frames in passing_frames_dict.items():
        game_id, play_id = play
        play_data = passing_play_data[(passing_play_data['gameId'] == game_id) & (passing_play_data['playId'] == play_id)].iloc[0]

        visualization.plot_frame(play_frames, play_data, f'{game_id}_{play_id}_norm', zoom=True)

        # Get ball location
        ball_x, ball_y = play_frames[(play_frames['displayName'] == 'football') & (play_frames['event'] == 'ball_snap')].iloc[0][['x', 'y']] #play_frames.iloc[0][['x', 'y']]
        print('ball_coords:', ball_x, ball_y)

        # Get QB location
        qb = play_frames[play_frames['nflId'].isin(all_qbs)].iloc[0]
        qb_x, qb_y = qb[['x', 'y']]
        qb_display = qb['displayName']
        print(qb_display, qb_x, qb_y)

        print('PLAY:\n', play_data)

        # Normalize the ball and all players to center of field
        # centered_play_frames = data_processing.normalize_to_center(play_frames, (ball_x, ball_y))
        # data_processing.plot_frame(centered_play_frames, play_data, f'{game_id}_{play_id}_norm_centered')

        # data_processing.detect_rushers(all_def_players, play_frames, (ball_x, ball_y), (qb_x, qb_y))



if __name__ == "__main__":
    main()