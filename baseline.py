import numpy as np
from sklearn.metrics import accuracy_score

# random SPSP values between 0-1
def random_probs(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random(n)


def confidence_accuracies(probs, y_true):

    # Masks for the three confidence categrories
    high_mask = probs >= 0.70
    med_mask  = (probs >= 0.40) & (probs < 0.70)
    low_mask  = probs < 0.40

    # 0/1 decision threshold (0.5) used for "hard" accuracy
    preds = (probs >= 0.5).astype(int)

    # Helper to compute accuracy on a mask, guarding against empty buckets
    def acc(mask):
        if mask.sum() == 0:
            return np.nan
        else:
            return accuracy_score(y_true[mask], preds[mask])

    return acc(high_mask), acc(med_mask), acc(low_mask)