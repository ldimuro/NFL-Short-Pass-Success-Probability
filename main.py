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
import torch
import random
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt


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

    passing_play_data, passing_tracking_data = data_processing.get_passing_plays(year=2022, week_start=1, week_end=1)

    print('passing play data:', len(passing_play_data))
    print('tracking data week #1:', len(passing_tracking_data[0]))

    if is_testing:
        passing_play_data = passing_play_data[passing_play_data['gameId'] <= 2022090800]

        frames = data_processing.get_pocket_frames(passing_play_data.iloc[[0]], passing_tracking_data)
        print(frames)




if __name__ == "__main__":
    main()