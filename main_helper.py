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
from models.cnn import cross_validation, BasicCNN, train_cnn
from models.logreg import cross_validation_lr
from models.baseline import confidence_accuracies, random_probs
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import brier_score_loss

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    check_random_state(seed_value)

all_player_data_2022 = get_data.get_player_data(year=2022)
all_player_data_2021 = get_data.get_player_data(year=2021)
all_player_data =  pd.concat([all_player_data_2022, all_player_data_2021])
all_players_data = all_player_data.drop_duplicates(subset=['nflId'])


def extract_short_passing_play_data():

    # Obtain all play and tracking data
    all_tracking_data_2021 = get_data.get_tracking_data(year=2021, week_start=1, week_end=8)
    all_tracking_data_2022 = get_data.get_tracking_data(year=2022, week_start=1, week_end=9)

    all_play_data_2021 = get_data.get_play_data(year=2021)
    all_play_data_2022 = get_data.get_play_data(year=2022)
    
    # Extract passing play data from 2021
    passing_play_data_2021 = all_play_data_2021[
        (all_play_data_2021['passResult'] == 'C') &
        (all_play_data_2021['penaltyYards'].isna()) &
        (all_play_data_2021['playDescription'].str.contains('short', case=False, na=False))
    ]
    # Extract tracking data
    passing_tracking_data_2021 = data_processing.filter_tracking_data(all_tracking_data_2021, passing_play_data_2021)
    passing_tracking_data_2021 = data_processing.normalize_field_direction(passing_tracking_data_2021)
    passing_tracking_data_2021 = data_processing.normalize_to_center(passing_tracking_data_2021)

    # Extract passing plays from 2022 (<= 2 yds past the LoS)
    passing_play_data_2022 = all_play_data_2022[all_play_data_2022['passResult'].notna()]
    short_pass_play_data_2022 = passing_play_data_2022[
        (passing_play_data_2022['passResult'] == 'C') & 
        (passing_play_data_2022['passLength'] <= 2) &
        (passing_play_data_2022['passTippedAtLine'] == False) &
        (passing_play_data_2022['playNullifiedByPenalty'] == 'N')
    ]
    # Extract tracking data
    passing_tracking_data_2022 = data_processing.filter_tracking_data(all_tracking_data_2022, short_pass_play_data_2022)
    passing_tracking_data_2022 = data_processing.normalize_field_direction(passing_tracking_data_2022)
    passing_tracking_data_2022 = data_processing.normalize_to_center(passing_tracking_data_2022)

    # Obtain all relevant data for 2021 plays
    short_pass_play_data_2021 = data_processing.get_data_at_pass_forward(passing_play_data_2021, passing_tracking_data_2021, all_players_data)
    print(f'PLAYS EXTRACTED {len(short_pass_play_data_2021)}/{len(passing_play_data_2021)}')
    data_processing.save_data(short_pass_play_data_2021, 'play_data/all_data/short_pass_play_data_2021_weeks1-8')
    # short_pass_play_data_2021_augmented = data_processing.augment_data(short_pass_play_data_2021)
    # data_processing.save_data(short_pass_play_data_2021_augmented, 'play_data/all_data/short_pass_play_data_2021_weeks1-8_augmented')

    # Obtain all relevant data for 2022 plays
    short_pass_play_data_2022 = data_processing.get_data_at_pass_forward(short_pass_play_data_2022, passing_tracking_data_2022, all_players_data)
    print(f'PLAYS EXTRACTED {len(short_pass_play_data_2022)}/{len(short_pass_play_data_2022)}')
    data_processing.save_data(short_pass_play_data_2022, 'play_data/all_data/short_pass_play_data_2022_weeks1-9')
    # short_pass_play_data_2022_augmented = data_processing.augment_data(short_pass_play_data_2022)
    # data_processing.save_data(short_pass_play_data_2022_augmented, 'short_pass_play_data_2022_weeks1-9_augmented')

    return passing_play_data_2021, passing_tracking_data_2021, short_pass_play_data_2022, passing_tracking_data_2022


def generate_input_tensors():
    total_data = get_saved_play_data()

    count_true = sum(1 for v in total_data.values() if v.get('label') is True)
    print(f'play success ratio: {count_true/len(total_data)*100:.2f}% ({count_true}/{len(total_data)})')

    # Create tensors/labels for all sample plays
    input_tensors, labels = data_processing.get_tensor_batch(total_data, all_players_data)
    data_processing.save_data(input_tensors, 'model_inputs/all_inputs/total_short_pass_input_tensors_dict')
    data_processing.save_data(labels, 'model_inputs/all_inputs/total_short_pass_labels_dict')


def create_withheld_testing_set(play_data_2021, play_tracking_2021, play_data_2022, play_tracking_2022):
    total_data = get_saved_play_data()

    input_tensors = data_processing.get_data('model_inputs/all_inputs/total_short_pass_input_tensors_dict')
    print('TOTAL INPUT TENSORS:', len(input_tensors))

    labels = data_processing.get_data('model_inputs/all_inputs/total_short_pass_labels_dict')
    print('TOTAL INPUT LABELS:', len(labels))

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

    withheld_plays = withheld_use_case_plays

    test_play_frames_data = {}
    for test_sample in withheld_plays:

        # Remove sample from the total dataset
        withheld_sample = total_data[test_sample]
        withheld_data = {}
        input_tensors.pop(test_sample, None)
        labels.pop(test_sample, None)

        withheld_data[test_sample] = withheld_sample

        # Extract every frame of the play
        test_game_id, test_play_id = test_sample
        if str(test_game_id).startswith('2021'):
            tracking_data = play_tracking_2021
            play_data = play_data_2021
        elif str(test_game_id).startswith('2022'):
            tracking_data = play_tracking_2022
            play_data = play_data_2022
        test_play_data = play_data[(play_data['gameId'] == test_game_id) & (play_data['playId'] == test_play_id)]

        # Extract frames between ball_snap and pass_forward
        test_play_frames = data_processing.get_relevant_frames(test_play_data, tracking_data, start_events=[constants.BALL_SNAP], end_events=[constants.PASS_FORWARD])

        # Add the play data for every frame of every test play
        z = test_play_frames[test_sample]
        min_frame = test_play_frames[test_sample]['frameId'].min()
        max_frame = test_play_frames[test_sample]['frameId'].max()
        for frame_id in range(min_frame, max_frame+1):
            data = withheld_data[test_sample].copy()
            data['tracking_data'] = z[z['frameId'] == frame_id]
            test_play_frames_data[(test_game_id, test_play_id+(frame_id*0.001))] = data


    data_processing.save_data(test_play_frames_data, 'play_data/test_data/test_short_pass_data')
    print('TOTAL DATA FOR TEST PLAYS:', len(test_play_frames_data))

    data_processing.save_data(input_tensors, 'model_inputs/training_set/train_short_pass_input_tensors')
    data_processing.save_data(labels, 'model_inputs/training_set/train_short_pass_labels')
    print(f'SAVED {len(input_tensors)} SAMPLES FOR TRAINING')

    test_input_tensors, test_labels = data_processing.get_tensor_batch(test_play_frames_data, all_players_data)
    data_processing.save_data(test_input_tensors, 'model_inputs/testing_set/test_short_pass_input_tensors')
    data_processing.save_data(test_labels, 'model_inputs/testing_set/test_short_pass_labels')
    print(f'SAVED {len(input_tensors)} SAMPLES FOR TESTING')


    # Ensure the testing sample is not in the training set
    for test_sample in withheld_plays:
        try:
            test = input_tensors[test_sample]
            print(f"FAIL: {test_sample} is in training set")
        except KeyError as e:
            print(f"PASS: {test_sample} is not in the training set")


def cross_validation_cnn(seeds=[42, 215, 23, 64]):
    x, y = get_training_data()

    # TRAIN CNN
    # seeds = [42, 215, 23, 64]

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # accumulators across seeds
    sum_pred = np.zeros(n_bins, dtype=np.float64)  # sum of predicted probs per bin
    sum_true = np.zeros(n_bins, dtype=np.float64)  # sum of true labels per bin
    count    = np.zeros(n_bins, dtype=np.int64)    # sample count per bin
    all_probs, all_trues = [], []                  # for pooled (micro) Brier

    # For High/Med/Low Confidence Accuracy bins
    def accumulate_bins(probs, trues):
        probs = np.asarray(probs, dtype=float).ravel()
        trues = np.asarray(trues, dtype=float).ravel()
        # bin index in [0, n_bins-1]
        idx = np.clip(np.digitize(probs, bin_edges, right=False) - 1, 0, n_bins - 1)
        # efficient in-place accumulation
        np.add.at(count,    idx, 1)
        np.add.at(sum_pred, idx, probs)
        np.add.at(sum_true, idx, trues)
        all_probs.append(probs)
        all_trues.append(trues)

    all_mean_acc = []
    all_pr_aucs = []
    all_roc_aucs = []
    all_brier = []
    all_hc = []
    all_mc = []
    all_lc = []
    for seed in seeds:
        set_seed(seed)

        pr_auc, roc_auc, brier, mean_hc, mean_mc, mean_lc, mean_acc, best_state, oof_trues, oof_preds = cross_validation(x, y, seed)
        all_pr_aucs.append(pr_auc)
        all_roc_aucs.append(roc_auc)
        all_brier.append(brier)
        all_hc.append(mean_hc)
        all_mc.append(mean_mc)
        all_lc.append(mean_lc)
        all_mean_acc.append(mean_acc)

        accumulate_bins(oof_preds, oof_trues)

    results = f"CNN TOTAL {len(seeds)}-SEED RESULTS:\n"
    results += f"MEAN PR-AUC: {np.mean(all_pr_aucs):.4f}\nMEAN ROC-AUC: {np.mean(all_roc_aucs):.4f}\nMEAN BRIER: {np.mean(all_brier):.4f}\n"
    results += f"MEAN HIGH CONF ACC: {np.mean(all_hc):.4f}\nMEAN MED CONF ACC: {np.mean(all_mc):.4f}\nMEAN LOW CONF ACC: {np.mean(all_lc):.4f}\n"
    results += f"MEAN ACC: {np.mean(all_mean_acc):.4f}"
    print(results)

    # FINAL POOLED CALIBRATION PLOT
    mask = count > 0
    avg_prob_pred = (sum_pred[mask] / count[mask])  # x: mean predicted prob per bin
    avg_prob_true = (sum_true[mask] / count[mask])  # y: empirical positive rate per bin

    # pooled (micro) Brier across all seeds
    micro_brier = brier_score_loss(np.concatenate(all_trues), np.concatenate(all_probs))
    print(f"Pooled (micro) Brier across seeds: {micro_brier:.4f}")

    plt.figure(figsize=(6, 5))
    plt.plot(avg_prob_pred, avg_prob_true, marker='o', label=f'CNN Avg Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Frequency', fontsize=12)
    plt.title('CNN Calibration Curve (pooled across 4 seeds)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/cnn_calibration_pooled.png', dpi=150)
    plt.close()



def cross_validation_logreg(seeds=[42, 215, 23, 64]):
    x, y = get_training_data()

    # TRAIN Logistic Regression
    all_mean_acc = []
    all_pr_aucs = []
    all_roc_aucs = []
    all_brier = []
    all_hc = []
    all_mc = []
    all_lc = []
    for seed in seeds:
        set_seed(seed)

        pr_auc, roc_auc, brier, mean_hc, mean_mc, mean_lc, mean_acc, best_state = cross_validation_lr(x, y, seed)
        all_pr_aucs.append(pr_auc)
        all_roc_aucs.append(roc_auc)
        all_brier.append(brier)
        all_hc.append(mean_hc)
        all_mc.append(mean_mc)
        all_lc.append(mean_lc)
        all_mean_acc.append(mean_acc)

    results = f"LOGISTIC REGRESSION TOTAL {len(seeds)}-SEED RESULTS:\n"
    results += f"MEAN PR-AUC: {np.mean(all_pr_aucs):.4f}\nMEAN ROC-AUC: {np.mean(all_roc_aucs):.4f}\nMEAN BRIER: {np.mean(all_brier):.4f}\n"
    results += f"MEAN HIGH CONF ACC: {np.mean(all_hc):.4f}\nMEAN MED CONF ACC: {np.mean(all_mc):.4f}\nMEAN LOW CONF ACC: {np.mean(all_lc):.4f}\n"
    results += f"MEAN ACC: {np.mean(all_mean_acc):.4f}"
    print(results)


def baseline_model(seeds=[42, 215, 23, 64]):
    x, y = get_training_data()

    # BASELINE CONFIDENCE ACCURACIES
    high_conf_accs = []
    med_conf_accs = []
    low_conf_accs = []
    for seed in seeds:
        probs = random_probs(len(y), seed)
        h_acc, m_acc, l_acc  = confidence_accuracies(probs, y)

        high_conf_accs.append(h_acc)
        med_conf_accs.append(m_acc)
        low_conf_accs.append(l_acc)

    mean_high_acc = np.mean(high_conf_accs)
    mean_med_acc = np.mean(med_conf_accs)
    mean_low_acc = np.mean(low_conf_accs)

    print('BASELINE CONFIDENCE ACCURACIES:')
    print(f"Mean High Conf Acc: {mean_high_acc:.4f}")
    print(f"Mean Med Conf Acc: {mean_med_acc:.4f}")
    print(f"Mean Low Conf Acc: {mean_low_acc:.4f}")


def train_final_cnn():
    x, y = get_training_data()
    set_seed()
    train_cnn(x, y)


def evaluate_test_set():
    test_input_tensors = data_processing.get_data('model_inputs/testing_set/test_short_pass_input_tensors')
    test_labels = data_processing.get_data('model_inputs/testing_set/test_short_pass_labels')
    all_test_play_data = data_processing.get_data('play_data/test_data/test_short_pass_data')

    # LOAD BEST STATE OF CNN
    test_model = BasicCNN()
    test_model.load_state_dict(torch.load('models/trained_cnn_model.pt', map_location='cpu'))
    test_model.eval()

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


                # Test accuracy of the first frame prediction
                if preds[0] == test_y.cpu().numpy()[0]:
                    first_frame_pred_correct += 1
                if preds[-1] == test_y.cpu().numpy()[0]:
                    last_frame_pred_correct += 1

                if test_y.cpu().numpy()[0] == 0:
                    if probs[0:3].sum()/3 <= 0.5:
                        first_rolling_avg_pred_correct += 1
                elif test_y.cpu().numpy()[0] == 1:
                    if probs[0:3].sum()/3 >= 0.5:
                        first_rolling_avg_pred_correct += 1

                # PRINT OUT FRAMES
                test_play_frames = pd.concat(test_play_frames, ignore_index=True)

                play_data = all_test_play_data[prev_frame]['play_data']

                # Get nflId of QB
                test_play_frames_with_pos = test_play_frames.merge(all_player_data[['nflId', 'position']], on='nflId', how='left')
                qb_id = test_play_frames_with_pos.query("position == 'QB'")['nflId'].unique().item()
                
                visualization.create_play_gif(play_data, test_play_frames, probs, all_test_play_data[prev_frame]['receiver_id'], qb_id, f'{prev_play[0]}_{prev_play[1]}', loop=True, zoom=False)


            print('new play:', game_id, play_id)
            total_samples += 1

            test_input_tensor_list = []
            test_label_list = []
            test_play_frames = []
            
        else:
            test_input_tensor_list.append(test_input_tensors[key])
            test_label_list.append(test_labels[key])
            test_play_frames.append(all_test_play_data[key]['tracking_data'])


        prev_play = (game_id, play_id)
        prev_frame = (game_id, play_id_frame)


    print(f'FIRST FRAME ACCURACY: {(first_frame_pred_correct/total_samples):.2f} ({first_frame_pred_correct}/{total_samples})')
    print(f'LAST FRAME ACCURACY: {(last_frame_pred_correct/total_samples):.2f} ({last_frame_pred_correct}/{total_samples})')
    print(f'FIRST ROLLING AVG ACCURACY: {(first_rolling_avg_pred_correct/total_samples):.2f} ({first_rolling_avg_pred_correct}/{total_samples})')


def get_saved_play_data():
    data_2021 = data_processing.get_data('play_data/all_data/short_pass_play_data_2021_weeks1-8') # 1142 samples
    data_2022 = data_processing.get_data('play_data/all_data/short_pass_play_data_2022_weeks1-9') # 1985 samples
    total_data = data_2021 | data_2022

    print('data_2021:', len(data_2021))
    print('data_2022:', len(data_2022))
    print('TOTAL DATA:', len(total_data))

    return total_data


def get_training_data():
    # Obtain training samples
    train_input_tensors = data_processing.get_data('model_inputs/training_set/train_short_pass_input_tensors')
    train_labels = data_processing.get_data('model_inputs/training_set/train_short_pass_labels')
    # test_labels = data_processing.get_data('test_behind_los_pass_input_tensors')

    print('# of training samples:', len(train_input_tensors))
    # print('# of testing samples:', len(test_labels))

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

    return x, y