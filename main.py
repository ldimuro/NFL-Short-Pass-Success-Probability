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
from cnn import cross_validation, BasicCNN, train_cnn
from logreg import cross_validation_lr
from baseline import confidence_accuracies, random_probs
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import brier_score_loss


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)


def main():
    print('running main')
    set_seed()


    # # --- Replace with your values (fractions 0..1) ---
    # categories = ['High Conf (70+% Success Pred)', 'Med Conf (40-69% Success Pred)', 'Low Conf (< 40% Success Pred)']

    # cnn = [0.8027, 0.5787, 0.7096]
    # lr  = [0.7131, 0.5765, 0.5589]
    # base= [0.5869, 0.5293, 0.3988]

    # # --- Plot ---
    # x = np.arange(len(categories))
    # width = 0.26

    # fig, ax = plt.subplots(figsize=(12, 6))

    # bars_cnn  = ax.bar(x - width, cnn,  width, label='CNN')
    # bars_lr   = ax.bar(x,         lr,   width, label='Logistic Regression')
    # bars_base = ax.bar(x + width, base, width, label='Baseline (random coin-flip)', color="#8B8B8B")

    # ax.set_xticks(x)
    # ax.set_xticklabels(categories, fontsize=14)
    # ax.set_ylim(0, 1)                            # data in 0..1
    # ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # ax.set_ylabel('Accuracy (%)', fontsize=18)
    # ax.tick_params(axis='y', labelsize=12)
    # ax.set_title('High/Med/Low Confidence Accuracy by Model', fontsize=22)
    # ax.grid(axis='y', linestyle='--', alpha=0.35)
    # ax.legend(fontsize=14)

    # # labels on top of bars
    # def add_labels(bars):
    #     for b in bars:
    #         h = b.get_height()
    #         ax.annotate(f'{h:.1%}',                 # e.g., 80.8%
    #                     xy=(b.get_x() + b.get_width()/2, h),
    #                     xytext=(0, 3), textcoords='offset points',
    #                     ha='center', va='bottom', fontsize=12)

    # for grp in (bars_cnn, bars_lr, bars_base):
    #     add_labels(grp)

    # plt.tight_layout()
    # plt.savefig('plots/confidences.png')









    is_training = True
    is_data_processing = False
    process_tensors = False
    is_test_set = False
    is_training_cnn = False
    is_evaluate_test_set = True

    all_player_data_2022 = get_data.get_player_data(year=2022)
    all_player_data_2021 = get_data.get_player_data(year=2021)
    all_player_data =  pd.concat([all_player_data_2022, all_player_data_2021])
    all_players_data = all_player_data.drop_duplicates(subset=['nflId'])


    if is_data_processing:

        # Obtain all play and tracking data
        all_tracking_data = get_data.get_tracking_data(year=2022, week_start=1, week_end=9)         # 9
        all_tracking_data_2021 = get_data.get_tracking_data(year=2021, week_start=1, week_end=8)    # 8
        all_tracking_data_2018 = get_data.get_tracking_data(year=2018, week_start=1, week_end=1)   # 17
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
                                                    (all_play_data_2021['penaltyYards'].isna()) &
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


    if is_training:

        data_2021 = data_processing.get_data('behind_los_play_data_2021_weeks1-8')  # 1142 samples
        data_2022 = data_processing.get_data('behind_los_play_data_2022_weeks1-9')  # 1985 samples
        data_2018 = data_processing.get_data('behind_los_play_data_2018_weeks1-17') # 4565 samples
        # data_2021_augm = data_processing.get_data('behind_los_play_data_2021_weeks1-8_augmented')
        # data_2022_augm = data_processing.get_data('behind_los_play_data_2022_weeks1-9_augmented')
        # data_2018_augm = data_processing.get_data('behind_los_play_data_2018_weeks1-17_augmented')
        total_data = data_2021 | data_2022 #| data_2018 #| data_2021_augm | data_2022_augm | data_2018_augm

        print('data_2021:', len(data_2021))
        print('data_2022:', len(data_2022))
        # print('data_2018:', len(data_2018))
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



        # Create tensors/labels for all sample plays
        if process_tensors:
            input_tensors, labels = data_processing.get_tensor_batch(total_data, all_players_data)
            data_processing.save_data(input_tensors, 'total_behind_los_pass_input_tensors_dict')
            data_processing.save_data(labels, 'total_behind_los_pass_labels_dict')

        

        input_tensors = data_processing.get_data('total_behind_los_pass_input_tensors_dict')
        print('TOTAL INPUT TENSORS:', len(input_tensors))
        labels = data_processing.get_data('total_behind_los_pass_labels_dict')
        print('TOTAL INPUT LABELS:', len(labels))





        if is_test_set:

            # - (2022091105, 2544) - No Success
            # - (2022091112, 917)  - Success
            # - (2021102403, 3496) - Success
            # - (2021100303, 1951) - No Success
            # - (2021102410, 3434) - No Success
            # - (2021091206, 3353) - MAIN EXAMPLE FOR VISUALIZATION, ADD INDICATOR FOR BOTH RECEIVERS (will need to manually add)
            withheld_use_case_plays = [
                (2021102404, 108), (2021091212, 611), (2022091112, 917), (2021100303, 1951), (2021102410, 3434),
                (2021091912, 3953), (2021091204, 2742), (2021100400, 262), (2022103008, 2713), (2022092509, 3717),
                (2022101606, 1414), (2021110100, 1351), (2022091105, 2544), (2021102405, 1665), (2021091206, 1171),
                (2022092900, 2204), (2022092200, 2589), (2021092605, 3769), (2021091909, 2392), (2022110609, 3668),
                (2021091203, 672), (2022101603, 2950), (2022100901, 2020), (2022102309, 2438), (2022103007, 1756),
                (2022102400, 1163), (2022100900, 3109), (2021103105, 4042), (2022091800, 3523), (2021091202, 3512),
                (2021092610, 3481), (2021101011, 1501), (2022100908, 2851), (2022091901, 1311), (2022102301, 1988),
                (2022102304, 1087), (2021100304, 484), (2022101605, 2054), (2022102700, 2820), (2021091202, 3536),
                (2022091101, 2951)
            ]

            # set_seed(44)

            # Randomly select 50 successful plays to use as testing set
            # success_plays = [key for key, value in total_data.items() if value.get('label') is True]
            # withheld_success_plays = random.sample(success_plays, 1) #100

            # # Randomly select 50 unsuccessful plays to use as testing set
            # no_success_plays = [key for key, value in total_data.items() if value.get('label') is False]
            # withheld_no_success_plays = random.sample(no_success_plays, 30) #100

            withheld_plays = withheld_use_case_plays#withheld_success_plays + withheld_no_success_plays + withheld_use_case_plays

            print('withheld_plays:', withheld_plays)
            

            # PERFECT EXAMPLE OF FAILURE PREDICTION: (2022091105, 2544)
            # PERFECT OVERALL EXAMPLE (USE THIS FOR DEMONSTRATION): (2021102404, 108)
            #   - #30 is not open at the beginning of the play, but as he drifts towards the sideline he becomes open enough to get a 1st down
            #   - The QB doesn't see him however, giving time for #29 to push in closer to #30, and by the time the QB throws, 
            #     #30's covered and doesn't get the 1st down
            # PERFECT EXAMPLE OF SUCCESS PREDICTION: (2022091112, 917)
            #   - The moment #33 passes by #55 moving in the opposite direction,
            #     the Short Pass Success Probability (SPSP) shoots up
            # (2021091212, 611) - Could've thrown it earlier for a higher SPSP, but threw it later and was unsuccessful

            # FALSE POSITIVE: (2021101704, 1613): Looks like #28 can easily get 2 yards but apparently he doesn't
            # FALSE NEGATIVE: (2021091911, 127): Looks like #86 has no chance for any gain, but in the actual scenario, he breaks a tackle for a big gain
            # FALSE POSITIVE: (2021091202, 3536): Great tackle by #40


            # (2021102405, 1665): Run this play with #1 as the intended receiver and compare SPSPs with #87




            test_play_frames_data = {}
            for test_sample in withheld_plays:

                # print('analyzing test play', test_sample)

                # REMOVE TEST SAMPLE:
                # test_sample = (2021102404, 108)#(2022091100, 458) #(2022091104,3204) #(2022091110, 514)
                # test_sample_aug = (2021092610, 1650.1)#(2022091100, 458.1)#(2022091104,3204.1)#(2022091110, 514.1)
                # test_sample = random.choice(list(data_2021.keys()))
                # test_sample_aug = (test_sample[0], test_sample[1]+0.1)

                # print(total_data[test_sample])
                withheld_sample = total_data[test_sample]
                # withheld_sample_aug = total_data[test_sample_aug]
                withheld_data = {}
                input_tensors.pop(test_sample, None)
                # input_tensors.pop(test_sample_aug, None)
                labels.pop(test_sample, None)
                # labels.pop(test_sample_aug, None)

                withheld_data[test_sample] = withheld_sample
                # withheld_data[test_sample_aug] = withheld_sample_aug
                # print('input_tensors mod:', len(input_tensors))
                # print('labels mod:', len(input_tensors))
                # print('WITHHELD DATA:', len(withheld_data))

                # Extract every frame of the play
                test_game_id, test_play_id = test_sample
                # print('TEST GAME ID:', test_game_id)
                if str(test_game_id).startswith('2018'):
                    tracking_data = passing_tracking_data_2018
                    play_data = passing_play_data_2018
                    # print('selected 2018 data')
                elif str(test_game_id).startswith('2021'):
                    tracking_data = passing_tracking_data_2021
                    play_data = passing_play_data_2021
                    # print('selected 2021 data')
                elif str(test_game_id).startswith('2022'):
                    tracking_data = passes_behind_los_tracking_data
                    play_data = passes_behind_los_play_data
                    # print('selected 2022 data')
                test_play_data = play_data[(play_data['gameId'] == test_game_id) & (play_data['playId'] == test_play_id)]

                
                # print(test_play_data)

                
                test_play_frames = data_processing.get_relevant_frames(test_play_data, tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_FORWARD])
                # print(test_play_frames[test_sample])

                # test_play_frames_data = {}
                # print('TEST:\n', test_play_frames)
                z = test_play_frames[test_sample]
                min_frame = test_play_frames[test_sample]['frameId'].min()
                max_frame = test_play_frames[test_sample]['frameId'].max()
                # print(f"Min:{min_frame}, Max:{max_frame}")
                for frame_id in range(min_frame, max_frame+1):
                    data = withheld_data[test_sample].copy()
                    data['tracking_data'] = z[z['frameId'] == frame_id]
                    test_play_frames_data[(test_game_id, test_play_id+(frame_id*0.001))] = data




            print('TOTAL DATA FOR TEST PLAYS:', len(test_play_frames_data))
            data_processing.save_data(test_play_frames_data, 'test_behind_los_pass_data')
            # print('TOTAL DATA FOR PLAY:', test_play_frames_data)

            data_processing.save_data(input_tensors, 'train_behind_los_pass_input_tensors')
            data_processing.save_data(labels, 'train_behind_los_pass_labels')
            print(f'SAVED {len(input_tensors)} SAMPLES FOR TRAINING')


            test_input_tensors, test_labels = data_processing.get_tensor_batch(test_play_frames_data, all_players_data)
            data_processing.save_data(test_input_tensors, 'test_behind_los_pass_input_tensors')
            data_processing.save_data(test_labels, 'test_behind_los_pass_labels')



            # Ensure the testing sample is not in the training set
            for test_sample in withheld_plays:
                try:
                    test = input_tensors[test_sample]
                    # test = input_tensors[test_sample_aug]
                    print(f"FAIL: {test_sample} is in training set")
                except KeyError as e:
                    print(f"PASS: {test_sample} is not in the training set")

        
        if is_training_cnn:
            # Obtain training samples
            train_input_tensors = data_processing.get_data('train_behind_los_pass_input_tensors')
            train_labels = data_processing.get_data('train_behind_los_pass_labels')
            test_labels = data_processing.get_data('test_behind_los_pass_input_tensors')

            print('# of training samples:', len(train_input_tensors))
            print('# of testing samples:', len(test_labels))

            # Convert test input tensors and labels from dict to list
            train_input_tensor_list = []
            train_label_list = []
            for key in train_input_tensors:
                train_input_tensor_list.append(train_input_tensors[key])
                train_label_list.append(train_labels[key])

            x = torch.from_numpy(np.array(train_input_tensor_list, dtype=np.float32))
            y = torch.from_numpy(np.array(train_label_list, dtype=np.int64))
            print('x:', x.shape)
            print('y:', y.shape)

            set_seed()
            train_cnn(x, y)

            # # TRAIN CNN
            # seeds = [42, 215, 23, 64]

            # n_bins = 10
            # bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

            # # accumulators across seeds
            # sum_pred = np.zeros(n_bins, dtype=np.float64)  # sum of predicted probs per bin
            # sum_true = np.zeros(n_bins, dtype=np.float64)  # sum of true labels per bin
            # count    = np.zeros(n_bins, dtype=np.int64)    # sample count per bin
            # all_probs, all_trues = [], []                  # for pooled (micro) Brier

            # def accumulate_bins(probs, trues):
            #     """Accumulate one seed's OOF predictions/labels into uniform bins."""
            #     probs = np.asarray(probs, dtype=float).ravel()
            #     trues = np.asarray(trues, dtype=float).ravel()
            #     # bin index in [0, n_bins-1]
            #     idx = np.clip(np.digitize(probs, bin_edges, right=False) - 1, 0, n_bins - 1)
            #     # efficient in-place accumulation
            #     np.add.at(count,    idx, 1)
            #     np.add.at(sum_pred, idx, probs)
            #     np.add.at(sum_true, idx, trues)
            #     all_probs.append(probs)
            #     all_trues.append(trues)

            # all_mean_acc = []
            # all_pr_aucs = []
            # all_roc_aucs = []
            # all_brier = []
            # all_hc = []
            # all_mc = []
            # all_lc = []
            # for seed in seeds:
            #     set_seed(seed)

            #     pr_auc, roc_auc, brier, mean_hc, mean_mc, mean_lc, mean_acc, best_state, oof_trues, oof_preds = cross_validation(x, y, seed)
            #     all_pr_aucs.append(pr_auc)
            #     all_roc_aucs.append(roc_auc)
            #     all_brier.append(brier)
            #     all_hc.append(mean_hc)
            #     all_mc.append(mean_mc)
            #     all_lc.append(mean_lc)
            #     all_mean_acc.append(mean_acc)

            #     accumulate_bins(oof_preds, oof_trues)

            # results = f"CNN TOTAL {len(seeds)}-SEED RESULTS:\n"
            # results += f"MEAN PR-AUC: {np.mean(all_pr_aucs):.4f}\nMEAN ROC-AUC: {np.mean(all_roc_aucs):.4f}\nMEAN BRIER: {np.mean(all_brier):.4f}\n"
            # results += f"MEAN HIGH CONF ACC: {np.mean(all_hc):.4f}\nMEAN MED CONF ACC: {np.mean(all_mc):.4f}\nMEAN LOW CONF ACC: {np.mean(all_lc):.4f}\n"
            # results += f"MEAN ACC: {np.mean(all_mean_acc):.4f}"
            # print(results)

            # # ===========================
            # # FINAL POOLED CALIBRATION PLOT
            # # ===========================
            # mask = count > 0
            # avg_prob_pred = (sum_pred[mask] / count[mask])  # x: mean predicted prob per bin
            # avg_prob_true = (sum_true[mask] / count[mask])  # y: empirical positive rate per bin

            # # pooled (micro) Brier across all seeds
            # micro_brier = brier_score_loss(np.concatenate(all_trues), np.concatenate(all_probs))
            # print(f"Pooled (micro) Brier across seeds: {micro_brier:.4f}")

            # plt.figure(figsize=(6, 5))
            # plt.plot(avg_prob_pred, avg_prob_true, marker='o', label=f'CNN Avg Calibration')
            # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
            # plt.xlabel('Predicted Probability', fontsize=12)
            # plt.ylabel('True Frequency', fontsize=12)
            # plt.title('CNN Calibration Curve (pooled across 4 seeds)', fontsize=14)
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig('plots/cnn_calibration_pooled.png', dpi=150)
            # plt.close()





            # # TRAIN Logistic Regression
            # all_mean_acc = []
            # all_pr_aucs = []
            # all_roc_aucs = []
            # all_brier = []
            # all_hc = []
            # all_mc = []
            # all_lc = []
            # for seed in seeds:
            #     set_seed(seed)

            #     pr_auc, roc_auc, brier, mean_hc, mean_mc, mean_lc, mean_acc, best_state = cross_validation_lr(x, y, seed)
            #     all_pr_aucs.append(pr_auc)
            #     all_roc_aucs.append(roc_auc)
            #     all_brier.append(brier)
            #     all_hc.append(mean_hc)
            #     all_mc.append(mean_mc)
            #     all_lc.append(mean_lc)
            #     all_mean_acc.append(mean_acc)

            # results = f"LOGISTIC REGRESSION TOTAL {len(seeds)}-SEED RESULTS:\n"
            # results += f"MEAN PR-AUC: {np.mean(all_pr_aucs):.4f}\nMEAN ROC-AUC: {np.mean(all_roc_aucs):.4f}\nMEAN BRIER: {np.mean(all_brier):.4f}\n"
            # results += f"MEAN HIGH CONF ACC: {np.mean(all_hc):.4f}\nMEAN MED CONF ACC: {np.mean(all_mc):.4f}\nMEAN LOW CONF ACC: {np.mean(all_lc):.4f}\n"
            # results += f"MEAN ACC: {np.mean(all_mean_acc):.4f}"
            # print(results)






            # # BASELINE CONFIDENCE ACCURACIES
            # high_conf_accs = []
            # med_conf_accs = []
            # low_conf_accs = []
            # for seed in seeds:
            #     probs = random_probs(len(y), seed)
            #     h_acc, m_acc, l_acc  = confidence_accuracies(probs, y)

            #     high_conf_accs.append(h_acc)
            #     med_conf_accs.append(m_acc)
            #     low_conf_accs.append(l_acc)

            # mean_high_acc = np.mean(high_conf_accs)
            # mean_med_acc = np.mean(med_conf_accs)
            # mean_low_acc = np.mean(low_conf_accs)

            # print('BASELINE CONFIDENCE ACCURACIES:')
            # print(f"Mean High Conf Acc: {mean_high_acc:.4f}")
            # print(f"Mean Med Conf Acc: {mean_med_acc:.4f}")
            # print(f"Mean Low Conf Acc: {mean_low_acc:.4f}")

            



        if is_evaluate_test_set:
            test_input_tensors = data_processing.get_data('test_behind_los_pass_input_tensors')
            test_labels = data_processing.get_data('test_behind_los_pass_labels')
            all_test_play_data = data_processing.get_data('test_behind_los_pass_data')
            train_samples = data_processing.get_data('train_behind_los_pass_input_tensors')
            # data_2021 = data_processing.get_data('behind_los_play_data_2021_weeks1-8')

            # print('HERE:\n', test_input_tensors.keys())

            # # LOAD BEST STATE OF CNN
            test_model = BasicCNN()
            # test_model.load_state_dict(torch.load('behind_los_success_withheld_model_test.pt', map_location='cpu'))
            test_model.load_state_dict(torch.load('best_model_withheld.pt', map_location='cpu'))
            test_model.eval()


            # for play,play_data in test_input_tensors.items():
            #     game_id, play_id = play
            #     play_id = int(str(play_id).split('.')[0]) # remove the frame number from the play_id
            #     print('play:', game_id, play_id)

            # Convert test input tensors and labels from dict to list
            test_input_tensor_list = []
            test_label_list = []
            test_play_frames = []
            prev_play = None
            prev_frame = None
            first_frame_pred_correct = 0
            last_frame_pred_correct = 0
            first_rolling_avg_pred_correct = 0
            total_samples = 0
            
            for key in test_input_tensors:
                game_id, play_id_frame = key
                play_id = int(str(play_id_frame).split('.')[0]) # remove the frame number from the play_id

                # print('play:', game_id, play_id)

                if prev_play != (game_id, play_id):
                    print('\tprev tensors:', len(test_input_tensor_list))

                    if len(test_input_tensor_list) > 0:
                        test_x = torch.from_numpy(np.array(test_input_tensor_list, dtype=np.float32))
                        test_y = torch.from_numpy(np.array(test_label_list, dtype=np.float32))
                        print('\ttest_x:', test_x.shape)
                        print('\ttest_y:', test_y.shape)

                        test_dataset   = TensorDataset(test_x, test_y)
                        test_loader    = DataLoader(test_dataset, batch_size=256)
                        with torch.no_grad():
                            logits = test_model(test_x).cpu().numpy()
                            probs  = 1 / (1 + np.exp(-logits))
                            print('\tPROBS:', len(probs))
                            preds  = (probs > 0.5).astype(int)


                        test_acc = np.mean(preds == test_y.cpu().numpy())
                        print('\tHold-out accuracy:', test_acc)



                        # try:
                        #     test = train_samples[(game_id, play_id)]
                        #     # test = input_tensors[(game_id, play_id)_aug]
                        #     print(f"FAIL: {(game_id, play_id)} is in training set")
                        # except KeyError as e:
                        #     print(f"PASS: {(game_id, play_id)} is not in the training set")



    

                        # Test accuracy of the first frame prediction
                        if preds[0] == test_y.cpu().numpy()[0]:
                            # print('CORRECT FIRST FRAME')
                            first_frame_pred_correct += 1
                        if preds[-1] == test_y.cpu().numpy()[0]:
                            # print('CORRECT LAST FRAME')
                            last_frame_pred_correct += 1

                        if test_y.cpu().numpy()[0] == 0:
                            if probs[0:3].sum()/3 <= 0.5:
                                first_rolling_avg_pred_correct += 1
                        elif test_y.cpu().numpy()[0] == 1:
                            if probs[0:3].sum()/3 >= 0.5:
                                first_rolling_avg_pred_correct += 1

                        # PRINT OUT FRAMES
                        test_play_frames = pd.concat(test_play_frames, ignore_index=True)
                        # print('TEST:\n', test_play_frames)
                        # print('PREV PLAY:', prev_frame)

                        play_data = all_test_play_data[prev_frame]['play_data']
                        # print('play_data:', play_data)

                        # Get nflId of QB
                        test_play_frames_with_pos = test_play_frames.merge(all_player_data[['nflId', 'position']], on='nflId', how='left')
                        qb_id = test_play_frames_with_pos.query("position == 'QB'")['nflId'].unique().item()
                        
                        visualization.create_play_gif(play_data, test_play_frames, probs, all_test_play_data[prev_frame]['receiver_id'], qb_id, f'{prev_play[0]}_{prev_play[1]}', loop=True, zoom=False)

                        # frame = test_play_frames[test_play_frames['frameId'] == test_play_frames['frameId'].min()]
                        # visualization.plot_frame_simple(frame, play_data, 0.7, 0.7, all_test_play_data[prev_frame]['receiver_id'], 'viz_test', zoom=False)

                        # print(data_2021[(2021103110, 99)])
                        # print(data_2021[(2021092600, 370)])
                        # print(data_2021[(2021101012, 3695)])

                        # break

                    # print(f"({game_id}, {play_id}),")

                    print('new play:', game_id, play_id)
                    total_samples += 1

                    test_input_tensor_list = []
                    test_label_list = []
                    test_play_frames = []

                    # if len(test_input_tensor_list) > 0:
                    #     break
                    
                else:
                    test_input_tensor_list.append(test_input_tensors[key])
                    test_label_list.append(test_labels[key])
                    test_play_frames.append(all_test_play_data[key]['tracking_data'])

                    # test_x = torch.from_numpy(np.array(test_input_tensor_list, dtype=np.float32))
                    # test_y = torch.from_numpy(np.array(test_label_list, dtype=np.float32))
                    # print('test_x:', test_x.shape)
                    # print('test_y:', test_y.shape)

                    # test_input_tensor_list = []
                    # test_label_list = []



                prev_play = (game_id, play_id)
                prev_frame = (game_id, play_id_frame)

                # test_input_tensor_list.append(test_input_tensors[key])
                # test_label_list.append(test_labels[key])

            print(f'FIRST FRAME ACCURACY: {(first_frame_pred_correct/total_samples):.2f} ({first_frame_pred_correct}/{total_samples})')
            print(f'LAST FRAME ACCURACY: {(last_frame_pred_correct/total_samples):.2f} ({last_frame_pred_correct}/{total_samples})')
            print(f'FIRST ROLLING AVG ACCURACY: {(first_rolling_avg_pred_correct/total_samples):.2f} ({first_rolling_avg_pred_correct}/{total_samples})')





if __name__ == "__main__":
    main()