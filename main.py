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

    # tracking_data = data_processing.get_tracking_data(year=2022, week_start=1, week_end=1)
    # print('TOTAL WEEKS OF TRACKING DATA:', len(tracking_data))

    play_data = data_processing.get_play_data(year=2022, pass_only=True)
    print(len(play_data))
    print(play_data.head())




if __name__ == "__main__":
    main()