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
import pickle


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
    is_data_processing = False

    all_player_data = get_data.get_player_data(year=2022)
    all_player_data_2021 = get_data.get_player_data(year=2021)


    if is_data_processing:

        # Obtain all play and tracking data
        all_tracking_data = get_data.get_tracking_data(year=2022, week_start=1, week_end=1)
        all_tracking_data_2021 = get_data.get_tracking_data(year=2021, week_start=1, week_end=1)
        all_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
        all_tracking_df_2021 = pd.concat(all_tracking_data_2021, ignore_index=True)
        all_play_data = get_data.get_play_data(year=2022)
        all_play_data_2021 = get_data.get_play_data(year=2021)
        # all_player_play_data = get_data.get_player_play_data(year=2022)

        

        # MAIN EXAMPLE: in (2021091206, 3353), 81 has a higher potential for yards, but QB throws to 28 instead
        # (2022091104, 3956): Goff could've passed it to 14 earlier and gotten a much larger gain
        # Good example (2022091104, 3204), (2022091100, 458), (2022091105, 4905), (2022091109, 743), (2022091112, 917)
        passing_play_data_2021 = all_play_data_2021[(all_play_data_2021['passResult'] == 'C') &
                                                    (all_play_data_2021['playDescription'].str.contains('short', case=False, na=False))]# & (all_play_data_2021['playResult'] <= 3)]
        passing_tracking_data_2021 = data_processing.filter_tracking_data(all_tracking_data_2021, passing_play_data_2021)
        passing_tracking_data_2021 = data_processing.normalize_field_direction(passing_tracking_data_2021)
        passing_tracking_data_2021 = data_processing.normalize_to_center(passing_tracking_data_2021)

        
        # Filter to include only pass plays that were thrown within 1 yards of the LoS
        passing_play_data = all_play_data[all_play_data['passResult'].notna()]
        passes_behind_los_play_data = passing_play_data[(passing_play_data['passResult'] == 'C') & 
                                                        (passing_play_data['passLength'] <= 2) &
                                                        (passing_play_data['passTippedAtLine'] == False) &
                                                        (passing_play_data['playNullifiedByPenalty'] == 'N')]# &
                                                        # (passing_play_data['targetY'] >= constants.SIDELINE_TO_HASH / 2) &
                                                        # (passing_play_data['targetY'] < constants.FIELD_WIDTH - constants.SIDELINE_TO_HASH / 2)]
        passes_behind_los_tracking_data = data_processing.filter_tracking_data(all_tracking_data, passes_behind_los_play_data)
        passes_behind_los_tracking_data = data_processing.normalize_field_direction(passes_behind_los_tracking_data)
        passes_behind_los_tracking_data = data_processing.normalize_to_center(passes_behind_los_tracking_data)
        

        print('# of passing plays:', len(passing_play_data))
        print('# of passing plays behind LoS:', len(passes_behind_los_play_data))
        print('#\t Average EPA on passes behind LoS:' ,passes_behind_los_play_data['expectedPointsAdded'].mean())

        median_yardsGained_yardsToGo_ratio = (passes_behind_los_play_data['yardsGained'] / passes_behind_los_play_data['yardsToGo']).median()
        plays_above_yardsGained_yardsToGo_ratio = passes_behind_los_play_data[(passes_behind_los_play_data['yardsGained']/passes_behind_los_play_data['yardsToGo']) >= median_yardsGained_yardsToGo_ratio]
        print('#\t Mean yardsGained/yardsToGo ratio on passes behind LoS:', median_yardsGained_yardsToGo_ratio)
        print('#\t Percent of behind LoS passes >= median_yardsGained_yardsToGo_ratio:', len(plays_above_yardsGained_yardsToGo_ratio) / len(passes_behind_los_play_data))
        print('#\t Max yardsGained on passes behind LoS:', passes_behind_los_play_data['yardsGained'].max())
        print('# of 2021 passing plays:', len(all_play_data_2021))


        # behind_los_play_data_2021 = data_processing.get_data_at_pass_forward(passing_play_data_2021, passing_tracking_data_2021, all_player_data_2021)
        # print(f'PLAYS EXTRACTED {len(behind_los_play_data_2021)}/{len(passing_play_data_2021)}')
        # data_processing.save_dict(behind_los_play_data_2021, 'behind_los_play_data_2021_weeks1-8')

        # behind_los_play_data_2022 = data_processing.get_data_at_pass_forward(passes_behind_los_play_data, passes_behind_los_tracking_data, all_player_data)
        # print(f'PLAYS EXTRACTED {len(behind_los_play_data_2022)}/{len(passes_behind_los_play_data)}')
        # data_processing.save_dict(behind_los_play_data_2022, 'behind_los_play_data_2022_weeks1-9')



    data_2021 = data_processing.get_dict('behind_los_play_data_2021_centered_weeks1-8')  # 1142 samples
    data_2022 = data_processing.get_dict('behind_los_play_data_2022_centered_weeks1-9')  # 1985 samples
    total_data = data_2021 | data_2022

    print('data_2021:', len(data_2021))
    print('data_2022:', len(data_2022))
    print('TOTAL DATA:', len(total_data))

    all_players =  pd.concat([all_player_data, all_player_data_2021])
    all_players = all_players.drop_duplicates(subset=['nflId'])
    count = 0
    for play,play_data in total_data.items():
        tensor = data_processing.create_input_tensor(play, play_data, all_players)
        count += 1

        if count == 1:
            break











    if is_data_processing:
        random_gameId, random_playId = random.choice(list(data_2021.keys()))
        test_data = passing_play_data_2021[(passing_play_data_2021['gameId'] == random_gameId) & (passing_play_data_2021['playId'] == random_playId)]
        passes_2021_dict = data_processing.get_relevant_frames(test_data, passing_tracking_data_2021, start_events=[constants.START], end_events=[constants.END])

        for play,play_frames in passes_2021_dict.items():
            game_id, play_id = play
            play_data = passing_play_data_2021[(passing_play_data_2021['gameId'] == game_id) & (passing_play_data_2021['playId'] == play_id)].iloc[0]
            print(play_frames)
            visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm_centered', loop=False, zoom=False)

        for i in range(1):
            random_gameId, random_playId = random.choice(list(data_2022.keys()))
            test_data = passes_behind_los_play_data[(passes_behind_los_play_data['gameId'] == random_gameId) & (passes_behind_los_play_data['playId'] == random_playId)]
            passes_2022_dict = data_processing.get_relevant_frames(test_data, passes_behind_los_tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_ARRIVED])

            for play,play_frames in passes_2022_dict.items():
                game_id, play_id = play
                play_data = passes_behind_los_play_data[(passes_behind_los_play_data['gameId'] == game_id) & (passes_behind_los_play_data['playId'] == play_id)].iloc[0]
                print(play_frames)
                visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm_centered', loop=False, zoom=False)



    

    # passes_behind_los_play_data = passes_behind_los_play_data[passes_behind_los_play_data['gameId'] <= 2022091200] # Week 1 only
    # test_passes_behind_los_plays = random.sample(range(len(passes_behind_los_play_data)), sample_num)
    # passes_behind_los_frames_dict = data_processing.get_relevant_frames(passes_behind_los_play_data.iloc[test_passes_behind_los_plays], passes_behind_los_tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_ARRIVED])

    # passing_play_data_2021 = passing_play_data_2021[(passing_play_data_2021['gameId'] <= 2021091300)] # Week 1 only
    # test_pass_plays_2021 = random.sample(range(len(passing_play_data_2021)), sample_num)
    # passes_2021_dict = data_processing.get_relevant_frames(passing_play_data_2021.iloc[test_pass_plays_2021], all_tracking_data_2021, start_events=[constants.BALL_SNAP], end_events=[constants.END])



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
    # passes_behind_los_success_labels = data_processing.estimate_play_success(passes_behind_los_play_data, year=2022)

    
    # passes_behind_los_tracking_data = pd.concat(passes_behind_los_tracking_data, ignore_index=True)
    # passing_tracking_data_2021 = pd.concat(passing_tracking_data_2021, ignore_index=True)
    # print('analyzing intended receivers')

    # Extract first and last name of receiver of a play in the playDescription and create 2 new columns for the name
    # passing_play_data_2021[['receiver_first_initial', 'receiver_last_name']] = passing_play_data_2021['playDescription'].apply(lambda desc: pd.Series(data_processing.extract_first_and_last_name(desc)))

    


    # with open('2021_behind_los_plays_weeks1-8.pkl', 'wb') as f:
    #     pickle.dump(behind_los_pass_plays_2021, f)

    # with open('2021_behind_los_plays_weeks1-8.pkl', 'rb') as f:
    #     behind_los_pass_plays_2021 = pickle.load(f)
    # print('got data')


    













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