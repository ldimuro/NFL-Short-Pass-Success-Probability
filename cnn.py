import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, 1), 
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // r, in_channels, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.excitation(self.squeeze(x))
        return x * scale
    
class EarlyStopper:
    def __init__(self, patience=8, mode='max'):
        self.patience = patience
        self.counter = 0
        self.best = -float('inf') if mode == 'max' else float('inf')
        self.mode = mode
        self.early_stop = False
        self.best_state = None


    def __call__(self, metric, model):
        if (metric > self.best) if self.mode == 'max' else (metric < self.best):
            self.best = metric
            self.counter = 0
            self.best_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



class BasicCNN(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(),

            SqueezeExcite(64, r=4),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )


    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze(-1)
    



def train_cnn(x, y, num_epochs=47):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10, random_state=42, stratify=y
    )
    
    # Datasets/Dataloaders
    train_data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_test, y_test)
    train_dataloader = DataLoader(train_data, batch_size=128)    #128
    val_dataloader = DataLoader(val_data, batch_size=256)        #256

    ## Model/Loss/Optimizer
    model = BasicCNN(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) # best so far: 1e-4
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - 0.59) / 0.59)) # handle class imbalance, since SUCCESS label represents 59% of data
    scheduler = OneCycleLR(optimizer, max_lr=3e-4, epochs=100, steps_per_epoch=len(train_dataloader)) # best so far: 3e-4

    best_val_pr_auc = 0.0
    best_state = None

    train_losses, val_losses = [], []
    # train_accs = [], []
    val_pr_aucs = []

    early_stop = EarlyStopper(patience=8, mode='max')

    epochs = num_epochs
    model.train()
    for epoch in range(epochs):
        # print('\tEPOCH', epoch+1)
        total_loss = 0
        correct, total = 0, 0

        for xb,yb in train_dataloader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(model(xb), yb.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            spsp = torch.sigmoid(output).squeeze()  # 'Short Pass Success Probability' - THE MAIN METRIC OF THE PROJECT

            # Binary Measure of Success
            pred = (spsp > 0.5).float()
            yb = yb.squeeze()

            # Overall Accuracy
            correct += (pred == yb).sum().item()
            total += yb.size(0)


        train_losses.append(total_loss / len(train_dataloader))
        # train_accs.append(correct / total)

        model.eval()
        val_loss = 0.0
        val_preds, val_trues = [], []


        with torch.no_grad():
            for xb, yb in val_dataloader:
                out = model(xb)
                val_loss += criterion(out, yb.float()).item()
                val_preds.extend(torch.sigmoid(out).squeeze().cpu().numpy())
                val_trues.extend(yb.cpu().numpy())


        val_pr_auc = auc(*precision_recall_curve(val_trues, val_preds)[1::-1])

        val_losses.append(val_loss / len(val_dataloader))
        val_pr_aucs.append(val_pr_auc)


        print(f'Epoch {epoch+1:02d} | '
            f'Train Loss {total_loss/len(train_dataloader):.4f} | '
            f'Val Loss {val_loss/len(val_dataloader):.4f} | '
            f'Val PR-AUC {val_pr_auc:.4f}')
        
        early_stop(val_pr_auc, model)
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss')


    plt.subplot(1, 3, 2)
    plt.plot(val_pr_aucs)
    plt.title('Val PR-AUC')


    plt.subplot(1, 3, 3)
    spsp_true, spsp_pred = calibration_curve(val_trues, val_preds, n_bins=10)
    plt.plot(spsp_pred, spsp_true, marker='o'); plt.plot([0,1],[0,1],'--')
    plt.title('Calibration')
    plt.tight_layout(); 
    plt.savefig('training_curves.png'); 
    plt.close()

    model.load_state_dict(early_stop.best_state)
    torch.save(early_stop.best_state, 'best_model_withheld.pt')





def cross_validation(x, y, seed, num_epochs=35, k=5): #33
    cross_val = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    hc_scores, mc_scores, lc_scores = [], [], []

    all_train_losses, all_val_losses = [], []
    all_train_accs, all_val_accs = [], []
    all_roc_aucs, all_log_losses, all_pr_aucs = [], [], []
    all_spsp_preds, all_spsp_trues = [], [] # for calibration

    best_loss = float('inf')
    best_state = None

    for fold,(train_i, val_i) in enumerate(cross_val.split(x, y)):
        print('FOLD:', fold+1)

        # Datasets/Dataloaders
        train_data = TensorDataset(x[train_i], y[train_i])
        val_data = TensorDataset(x[val_i], y[val_i])
        train_dataloader = DataLoader(train_data, batch_size=128)    #128
        val_dataloader = DataLoader(val_data, batch_size=256)        #256

        # Model/Loss/Optimizer
        model = BasicCNN(in_channels=x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) # best so far: 1e-4
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - 0.59) / 0.59)) # handle class imbalance, since SUCCESS label represents 59% of data
        scheduler = OneCycleLR(optimizer, max_lr=3e-4, epochs=100, steps_per_epoch=len(train_dataloader)) # best so far: 3e-4

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        val_hc_accs = []
        val_mc_accs = []
        val_lc_accs = []

        epochs = num_epochs
        model.train()
        for epoch in range(epochs):
            print('\tEPOCH', epoch+1)
            total_loss = 0
            correct, total = 0, 0

            for xb,yb in train_dataloader:
                optimizer.zero_grad()
                output = model(xb)
                loss = criterion(model(xb), yb.float())
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                spsp = torch.sigmoid(output).squeeze()  # 'Short Pass Success Probability' - THE MAIN METRIC OF THE PROJECT

                # Binary Measure of Success
                pred = (spsp > 0.5).float()
                yb = yb.squeeze()

                # Overall Accuracy
                correct += (pred == yb).sum().item()
                total += yb.size(0)


            train_losses.append(total_loss / len(train_dataloader))
            train_accs.append(correct / total)
        
            # Validation eval
            model.eval()
            val_loss = 0
            # val_correct, val_hc_correct, val_lc_correct, val_total, val_hc_total, val_lc_total = 0, 0, 0, 0, 0, 0
            val_correct, val_total = 0, 0
            val_hc_correct, val_hc_total = 0, 0
            val_mc_correct, val_mc_total = 0, 0
            val_lc_correct, val_lc_total = 0, 0
            with torch.no_grad():
                for xb, yb in val_dataloader:
                    output = model(xb)
                    loss = criterion(output, yb.float())
                    val_loss += loss.item()

                    spsp = torch.sigmoid(output).squeeze()

                    # Binary Measure of Success
                    pred = (spsp > 0.5).float()
                    yb = yb.squeeze()

                    # Overall Accuracy
                    val_correct += (pred == yb).sum().item()
                    val_total += yb.size(0)

                    # High-Confidence Prediction Accuracy
                    high_conf_pred = (spsp >= 0.7)
                    if high_conf_pred.any():
                        val_hc_correct += (pred[high_conf_pred] == yb[high_conf_pred]).sum().item()
                        val_hc_total += high_conf_pred.sum().item()

                    # Med-Confidence Prediction Accuracy
                    med_conf_pred = (spsp >= 0.4) & (spsp < 0.7)
                    if med_conf_pred.any():
                        val_mc_correct += (pred[med_conf_pred] == yb[med_conf_pred]).sum().item()
                        val_mc_total += med_conf_pred.sum().item()

                    # Low-Confidence Prediction Accuracy
                    low_conf_pred = (spsp < 0.4)
                    if low_conf_pred.any():
                        val_lc_correct += (pred[low_conf_pred] == yb[low_conf_pred]).sum().item()
                        val_lc_total += low_conf_pred.sum().item()


            val_losses.append(val_loss / len(val_dataloader))
            val_accs.append(val_correct / val_total)
            val_hc_accs.append(val_hc_correct / (val_hc_total if val_hc_total > 0 else 1))
            val_mc_accs.append(val_mc_correct / (val_mc_total if val_mc_total > 0 else 1))
            val_lc_accs.append(val_lc_correct / (val_lc_total if val_lc_total > 0 else 1))

            # # Update best model across all folds
            # avg_val_loss = val_loss / len(val_loader)
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     best_state = model.state_dict()

        scores.append(val_accs[-1])
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        hc_scores.append(val_hc_accs[-1])
        mc_scores.append(val_mc_accs[-1])
        lc_scores.append(val_lc_accs[-1])

        
        spsp_preds, spsp_trues = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_dataloader:
                output = model(xb)
                spsp_pred = torch.sigmoid(output).squeeze().cpu().numpy()
                yb = yb.squeeze().cpu().numpy()

                spsp_preds.extend(spsp_pred.tolist())
                spsp_trues.extend(yb.tolist())

        # Save best model across folds
        avg_val_loss = val_loss / len(val_dataloader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = model.state_dict()

        # Save for Calibration Curve
        all_spsp_trues.extend(spsp_trues)
        all_spsp_preds.extend(spsp_preds)

        # Compute ROC-AUC and Log Loss for this fold
        roc = roc_auc_score(spsp_trues, spsp_preds)
        logloss = log_loss(spsp_trues, spsp_preds)

        precision, recall, _ = precision_recall_curve(spsp_trues, spsp_preds)
        pr_auc = auc(recall, precision)

        all_roc_aucs.append(roc)
        all_log_losses.append(logloss)
        all_pr_aucs.append(pr_auc)


        print(f'\tFold {fold+1} ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}, Log Loss: {logloss:.4f}')

    # Plot average over folds
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss = np.mean(all_val_losses, axis=0)
    avg_train_acc = np.mean(all_train_accs, axis=0)
    avg_val_acc = np.mean(all_val_accs, axis=0)


    # Calibration Curve
    spsp_true, spsp_pred = calibration_curve(all_spsp_trues, all_spsp_preds, n_bins=10, strategy='uniform')
    brier = brier_score_loss(all_spsp_trues, all_spsp_preds)
    baseline_preds = np.full(len(all_spsp_trues), 0.59)
    baseline_brier = brier_score_loss(all_spsp_trues, baseline_preds)
    print(f'\nBrier Score: {brier:.4f} (Baseline: {baseline_brier:.4f})')

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(avg_train_acc, label='Train Acc')
    plt.plot(avg_val_acc, label='Val Acc')
    plt.title(f'Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.subplot(1, 3, 2)
    # plt.plot(all_pr_aucs, label='Val PR-AUC')
    # plt.title(f'PR-AUC')
    # plt.xlabel('Epoch')
    # plt.ylabel('PR-AUC')
    # plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(spsp_true, spsp_pred, marker='o', label='Model Calibration')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Frequency')
    plt.title('Calibration Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_val_loss_accuracy.png')
    plt.close()

    oof_roc = roc_auc_score(all_spsp_trues, all_spsp_preds)
    oof_logloss = log_loss(all_spsp_trues, all_spsp_preds)
    precision, recall, _ = precision_recall_curve(all_spsp_trues, all_spsp_preds)
    oof_pr_auc = auc(recall, precision)
    print(f'OOF ROC-AUC: {oof_roc:.4f}')
    print(f'OOF PR-AUC: {oof_pr_auc:.4f}')
    print(f'OOF LogLoss: {oof_logloss:.4f}')

    # High/Med/Low Confidence Accuracy
    print(f'Avg High Conf Acc across folds: {np.mean(hc_scores)}')
    print(f'Avg Med Conf Acc across folds: {np.mean(mc_scores)}')
    print(f'Avg Low Conf Acc across folds: {np.mean(lc_scores)}')

    # print(f'Avg ROC-AUC across folds: {np.mean(all_roc_aucs):.4f} +/- {np.std(all_roc_aucs):.4f}')
    # print(f'Avg PR-AUC across folds: {np.mean(all_pr_aucs):.4f} +/- {np.std(all_pr_aucs):.4f}')
    # print(f'Avg Log Loss across folds: {np.mean(all_log_losses):.4f} +/- {np.std(all_log_losses):.4f}')


    # return np.mean(scores), np.std(scores), best_loss, best_state
    return oof_pr_auc, oof_roc, brier, np.mean(hc_scores), np.mean(mc_scores), np.mean(lc_scores), np.mean(scores), best_state

