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
from cnn import cross_validation


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)


def main():
    print('running main')
    set_seed()

    is_data_processing = False

    all_player_data = get_data.get_player_data(year=2022)
    all_player_data_2021 = get_data.get_player_data(year=2021)
    all_player_data_2018 = get_data.get_player_data(year=2018)



    if is_data_processing:

        # Obtain all play and tracking data
        all_tracking_data = get_data.get_tracking_data(year=2022, week_start=1, week_end=1)         # 9
        all_tracking_data_2021 = get_data.get_tracking_data(year=2021, week_start=1, week_end=1)    # 8
        all_tracking_data_2018 = get_data.get_tracking_data(year=2018, week_start=1, week_end=17)   # 17
        # all_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
        # all_tracking_df_2021 = pd.concat(all_tracking_data_2021, ignore_index=True)
        all_play_data = get_data.get_play_data(year=2022)
        all_play_data_2021 = get_data.get_play_data(year=2021)
        all_play_data_2018 = get_data.get_play_data(year=2018)
        # all_player_play_data = get_data.get_player_play_data(year=2022)

        

        # MAIN EXAMPLE: in (2021091206, 3353), 81 has a higher potential for yards, but QB throws to 28 instead
        # (2022091104, 3956): Goff could've passed it to 14 earlier and gotten a much larger gain
        # Good example (2022091104, 3204), (2022091100, 458), (2022091105, 4905), (2022091109, 743), (2022091112, 917)
        passing_play_data_2021 = all_play_data_2021[(all_play_data_2021['passResult'] == 'C') &
                                                    (all_play_data_2021['playDescription'].str.contains('short', case=False, na=False))]# & (all_play_data_2021['playResult'] <= 3)]
        passing_tracking_data_2021 = data_processing.filter_tracking_data(all_tracking_data_2021, passing_play_data_2021)
        passing_tracking_data_2021 = data_processing.normalize_field_direction(passing_tracking_data_2021)
        passing_tracking_data_2021 = data_processing.normalize_to_center(passing_tracking_data_2021)


        passing_play_data_2018 = all_play_data_2018[(all_play_data_2018['passResult'] == 'C') &
                                                    (all_play_data_2018['playDescription'].str.contains('short', case=False, na=False))]# & (all_play_data_2018['playResult'] <= 3)]
        passing_tracking_data_2018 = data_processing.filter_tracking_data(all_tracking_data_2018, passing_play_data_2018)
        passing_tracking_data_2018 = data_processing.normalize_field_direction(passing_tracking_data_2018)
        passing_tracking_data_2018 = data_processing.normalize_to_center(passing_tracking_data_2018)

        
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
        

        # print('# of passing plays:', len(passing_play_data))
        # print('# of passing plays behind LoS:', len(passes_behind_los_play_data))
        # print('#\t Average EPA on passes behind LoS:' ,passes_behind_los_play_data['expectedPointsAdded'].mean())

        # median_yardsGained_yardsToGo_ratio = (passes_behind_los_play_data['yardsGained'] / passes_behind_los_play_data['yardsToGo']).median()
        # plays_above_yardsGained_yardsToGo_ratio = passes_behind_los_play_data[(passes_behind_los_play_data['yardsGained']/passes_behind_los_play_data['yardsToGo']) >= median_yardsGained_yardsToGo_ratio]
        # print('#\t Mean yardsGained/yardsToGo ratio on passes behind LoS:', median_yardsGained_yardsToGo_ratio)
        # print('#\t Percent of behind LoS passes >= median_yardsGained_yardsToGo_ratio:', len(plays_above_yardsGained_yardsToGo_ratio) / len(passes_behind_los_play_data))
        # print('#\t Max yardsGained on passes behind LoS:', passes_behind_los_play_data['yardsGained'].max())
        # print('# of 2021 passing plays:', len(all_play_data_2021))


        # behind_los_play_data_2021 = data_processing.get_data_at_pass_forward(passing_play_data_2021, passing_tracking_data_2021, all_player_data_2021)
        # print(f'PLAYS EXTRACTED {len(behind_los_play_data_2021)}/{len(passing_play_data_2021)}')
        # data_processing.save_data(behind_los_play_data_2021, 'behind_los_play_data_2021_weeks1-8')
        # behind_los_play_data_2021_augmented = data_processing.augment_data(behind_los_play_data_2021)
        # data_processing.save_data(behind_los_play_data_2021_augmented, 'behind_los_play_data_2021_weeks1-8_augmented')

        # behind_los_play_data_2022 = data_processing.get_data_at_pass_forward(passes_behind_los_play_data, passes_behind_los_tracking_data, all_player_data)
        # print(f'PLAYS EXTRACTED {len(behind_los_play_data_2022)}/{len(passes_behind_los_play_data)}')
        # data_processing.save_data(behind_los_play_data_2022, 'behind_los_play_data_2022_weeks1-9')
        # behind_los_play_data_2022_augmented = data_processing.augment_data(behind_los_play_data_2022)
        # data_processing.save_data(behind_los_play_data_2022_augmented, 'behind_los_play_data_2022_weeks1-9_augmented')

        # behind_los_play_data_2018 = data_processing.get_data_at_pass_forward(passing_play_data_2018, passing_tracking_data_2018, all_player_data_2018)
        # print(f'PLAYS EXTRACTED {len(behind_los_play_data_2018)}/{len(passing_play_data_2018)}')
        # data_processing.save_data(behind_los_play_data_2018, 'behind_los_play_data_2018_weeks1-17')
        # behind_los_play_data_2018_augmented = data_processing.augment_data(behind_los_play_data_2018)
        # data_processing.save_data(behind_los_play_data_2018_augmented, 'behind_los_play_data_2018_weeks1-17_augmented')



    data_2021 = data_processing.get_data('behind_los_play_data_2021_weeks1-8')  # 1142 samples
    data_2022 = data_processing.get_data('behind_los_play_data_2022_weeks1-9')  # 1985 samples
    data_2018 = data_processing.get_data('behind_los_play_data_2018_weeks1-17') # 4565 samples
    data_2021_augm = data_processing.get_data('behind_los_play_data_2021_weeks1-8_augmented')
    data_2022_augm = data_processing.get_data('behind_los_play_data_2022_weeks1-9_augmented')
    data_2018_augm = data_processing.get_data('behind_los_play_data_2018_weeks1-17_augmented')
    total_data = data_2021 | data_2022 | data_2018 | data_2021_augm | data_2022_augm | data_2018_augm

    print('data_2021:', len(data_2021))
    print('data_2022:', len(data_2022))
    print('data_2018:', len(data_2018))
    print('TOTAL DATA:', len(total_data))

    # print('AUG:\n', list(data_2021_augm)[5], data_2021_augm[list(data_2021_augm)[5]])
    # print('NON-AUG:\n', list(data_2021)[5], data_2021[list(data_2021)[5]])

    # print('AUG:\n', list(data_2022_augm)[5], data_2022_augm[list(data_2022_augm)[5]])
    # print('NON-AUG:\n', list(data_2022)[5], data_2022[list(data_2022)[5]])

    # print('AUG:\n', list(data_2018_augm)[5], data_2018_augm[list(data_2018_augm)[5]])
    # print('NON-AUG:\n', list(data_2018)[5], data_2018[list(data_2018)[5]])

    # print(total_data[(2021092604,3981)])

    count_true = sum(1 for v in total_data.values() if v.get('label') is True)
    print(f'play success ratio: {count_true/len(total_data)*100:.2f}% ({count_true}/{len(total_data)})')




    # all_players =  pd.concat([all_player_data, all_player_data_2021, all_player_data_2018])
    # all_players = all_players.drop_duplicates(subset=['nflId'])
    # input_tensors = []
    # labels = []
    # skipped = []
    # for play,play_data in total_data.items():
    #     game_id, play_id = play

    #     # Occasionally there are more/less than 11 players on each side, catch this error and skip
    #     try:
    #         # Create input tensor
    #         tensor = data_processing.create_input_tensor(play_data, all_players)
    #         input_tensors.append(tensor)

    #         # Save corresponding label to input tensor
    #         label = int(play_data['label'])
    #         labels.append(label)

    #         print(f"created tensor+label for ({game_id},{play_id})")


    #     except:
    #         skipped.append(play)
    #         print(f"ERROR FOR ({game_id},{play_id})")

    # print('skipped:', len(skipped))
    # print('FINAL TENSOR COUNT:', len(input_tensors)) # 7683 total input tensors
    # print('FINAL LABEL COUNT:', len(labels))

    # data_processing.save_data(input_tensors, 'total_behind_los_pass_aug_input_tensors')
    # data_processing.save_data(labels, 'total_behind_los_pass_aug_labels')




    input_tensors = data_processing.get_data('total_behind_los_pass_aug_input_tensors')
    print('TOTAL INPUT TENSORS:', len(input_tensors))

    labels = data_processing.get_data('total_behind_los_pass_aug_labels')
    print('TOTAL INPUT LABELS:', len(labels))


    x = torch.from_numpy(np.array(input_tensors, dtype=np.float32))
    print('x:', x.shape)

    y = torch.from_numpy(np.array(labels, dtype=np.int64))
    print('y:', y.shape)


    mean_acc, std_acc = cross_validation(x, y)
    print(f"Mean Cross-Val accuracy: {mean_acc:.3f} += {std_acc:.3f}")









    # if is_data_processing:
        # random_gameId, random_playId = random.choice(list(data_2018.keys()))
        # test_data = passing_play_data_2018[(passing_play_data_2018['gameId'] == random_gameId) & (passing_play_data_2018['playId'] == random_playId)]
        # passes_2018_dict = data_processing.get_relevant_frames(test_data, passing_tracking_data_2018, start_events=[constants.START], end_events=[constants.END])

        # for play,play_frames in passes_2018_dict.items():
        #     game_id, play_id = play
        #     play_data = passing_play_data_2018[(passing_play_data_2018['gameId'] == game_id) & (passing_play_data_2018['playId'] == play_id)].iloc[0]
        #     print(play_frames)
        #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm_centered', loop=False, zoom=False)

        # random_gameId, random_playId = random.choice(list(data_2021.keys()))
        # test_data = passing_play_data_2021[(passing_play_data_2021['gameId'] == random_gameId) & (passing_play_data_2021['playId'] == random_playId)]
        # passes_2021_dict = data_processing.get_relevant_frames(test_data, passing_tracking_data_2021, start_events=[constants.START], end_events=[constants.END])

        # for play,play_frames in passes_2021_dict.items():
        #     game_id, play_id = play
        #     play_data = passing_play_data_2021[(passing_play_data_2021['gameId'] == game_id) & (passing_play_data_2021['playId'] == play_id)].iloc[0]
        #     print(play_frames)
        #     visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm_centered', loop=False, zoom=False)

        # for i in range(1):
        #     random_gameId, random_playId = random.choice(list(data_2018.keys()))
        #     test_data = passes_behind_los_play_data[(passes_behind_los_play_data['gameId'] == random_gameId) & (passes_behind_los_play_data['playId'] == random_playId)]
        #     passes_2022_dict = data_processing.get_relevant_frames(test_data, passes_behind_los_tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_ARRIVED])

        #     for play,play_frames in passes_2022_dict.items():
        #         game_id, play_id = play
        #         play_data = passes_behind_los_play_data[(passes_behind_los_play_data['gameId'] == game_id) & (passes_behind_los_play_data['playId'] == play_id)].iloc[0]
        #         print(play_frames)
        #         visualization.create_play_gif(play_data, play_frames, f'{game_id}_{play_id}_behind_los_norm_centered', loop=False, zoom=False)







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