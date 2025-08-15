import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# Logistic Regression on flattened (13,11,10)
class LogisticReg(nn.Module):
    def __init__(self, in_dim=13*11*10):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)


def cross_validation_lr(x, y, seed, num_epochs=4, k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cross_val = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    scores = []
    hc_scores, mc_scores, lc_scores = [], [], []

    all_train_losses, all_val_losses = [], []
    all_train_accs,  all_val_accs  = [], []
    all_roc_aucs, all_log_losses, all_pr_aucs = [], [], []
    all_spsp_preds, all_spsp_trues = [], []  # for calibration

    best_loss  = float('inf')
    best_state = None

    pos_rate = 0.59  # SUCCESS class percentage
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate, dtype=torch.float32, device=device)

    for fold, (train_i, val_i) in enumerate(cross_val.split(x, y)):
        print('FOLD:', fold + 1)

        # Datasets / Dataloaders
        train_data = TensorDataset(x[train_i], y[train_i])
        val_data   = TensorDataset(x[val_i],  y[val_i])
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        val_loader   = DataLoader(val_data,   batch_size=256, shuffle=False)

        # Model/Loss/Optimizer
        model = LogisticReg(in_dim=13*11*10).to(device)
        criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

        train_losses, val_losses = [], []
        train_accs,  val_accs   = [], []
        val_hc_accs, val_mc_accs, val_lc_accs = [], [], []

        # Train
        for epoch in range(num_epochs):
            print('\tEPOCH', epoch + 1)
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).float().view(-1, 1)

                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                spsp = torch.sigmoid(logits).squeeze(1)   # probabilities
                pred = (spsp > 0.5).float()
                correct += (pred == yb.squeeze(1)).sum().item()
                total   += yb.size(0)

            train_losses.append(running_loss / len(train_loader))
            train_accs.append(correct / total)

            # Validate
            model.eval()
            v_loss = 0.0
            val_correct, val_total = 0, 0
            val_hc_correct = val_hc_total = 0
            val_mc_correct = val_mc_total = 0
            val_lc_correct = val_lc_total = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device).float().view(-1, 1)

                    logits = model(xb)
                    loss = criterion(logits, yb)
                    v_loss += loss.item()

                    spsp = torch.sigmoid(logits).squeeze(1)
                    pred = (spsp > 0.5).float()
                    yb1  = yb.squeeze(1)

                    # Overall
                    val_correct += (pred == yb1).sum().item()
                    val_total   += yb1.size(0)

                    # Confidence bins
                    hc = (spsp >= 0.7)
                    mc = (spsp >= 0.4) & (spsp < 0.7)
                    lc = (spsp < 0.4)

                    if hc.any():
                        val_hc_correct += (pred[hc] == yb1[hc]).sum().item()
                        val_hc_total   += hc.sum().item()
                    if mc.any():
                        val_mc_correct += (pred[mc] == yb1[mc]).sum().item()
                        val_mc_total   += mc.sum().item()
                    if lc.any():
                        val_lc_correct += (pred[lc] == yb1[lc]).sum().item()
                        val_lc_total   += lc.sum().item()

            val_losses.append(v_loss / len(val_loader))
            val_accs.append(val_correct / val_total)
            val_hc_accs.append(val_hc_correct / (val_hc_total if val_hc_total > 0 else 1))
            val_mc_accs.append(val_mc_correct / (val_mc_total if val_mc_total > 0 else 1))
            val_lc_accs.append(val_lc_correct / (val_lc_total if val_lc_total > 0 else 1))

        # Store per-fold summaries
        scores.append(val_accs[-1])
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        hc_scores.append(val_hc_accs[-1])
        mc_scores.append(val_mc_accs[-1])
        lc_scores.append(val_lc_accs[-1])

        # Collect OOF preds for calibration/OOF metrics
        spsp_preds, spsp_trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                y_np = yb.cpu().numpy().astype(np.float32).reshape(-1)
                spsp_preds.extend(prob.tolist())
                spsp_trues.extend(y_np.tolist())

        # Track best by validation loss
        avg_val_loss = v_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # store on CPU

        # Per-fold metrics
        roc = roc_auc_score(spsp_trues, spsp_preds)
        logloss = log_loss(spsp_trues, spsp_preds)
        precision, recall, _ = precision_recall_curve(spsp_trues, spsp_preds)
        pr_auc = auc(recall, precision)

        all_roc_aucs.append(roc)
        all_log_losses.append(logloss)
        all_pr_aucs.append(pr_auc)

        # Add to global OOF lists
        all_spsp_trues.extend(spsp_trues)
        all_spsp_preds.extend(spsp_preds)

        print(f'\tFold {fold+1} ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}, Log Loss: {logloss:.4f}')

    # Averages across epochs (for plotting)
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_acc = np.mean(all_train_accs, axis=0)
    avg_val_acc = np.mean(all_val_accs, axis=0)

    # Calibration
    spsp_true_bins, spsp_pred_bins = calibration_curve(all_spsp_trues, all_spsp_preds, n_bins=10, strategy='uniform')
    brier = brier_score_loss(all_spsp_trues, all_spsp_preds)
    baseline_brier = brier_score_loss(all_spsp_trues, np.full(len(all_spsp_trues), pos_rate))
    print(f'\nBrier Score: {brier:.4f} (Baseline: {baseline_brier:.4f})')

    # Plots
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss,   label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(avg_train_acc, label='Train Acc')
    plt.plot(avg_val_acc,   label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(spsp_true_bins, spsp_pred_bins, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Frequency')
    plt.title('Calibration Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/train_val_loss_accuracy_logreg.png')
    plt.close()

    # Out-of-fold metrics (global)
    oof_roc = roc_auc_score(all_spsp_trues, all_spsp_preds)
    oof_logloss = log_loss(all_spsp_trues, all_spsp_preds)
    precision, recall, _ = precision_recall_curve(all_spsp_trues, all_spsp_preds)
    oof_pr_auc = auc(recall, precision)

    print(f'OOF ROC-AUC: {oof_roc:.4f}')
    print(f'OOF PR-AUC: {oof_pr_auc:.4f}')
    print(f'OOF LogLoss: {oof_logloss:.4f}')

    print(f'Avg High Conf Acc across folds: {np.mean(hc_scores):.4f}')
    print(f'Avg Med  Conf Acc across folds: {np.mean(mc_scores):.4f}')
    print(f'Avg Low  Conf Acc across folds: {np.mean(lc_scores):.4f}')

    return oof_pr_auc, oof_roc, brier, np.mean(hc_scores), np.mean(mc_scores), np.mean(lc_scores), np.mean(scores), best_state