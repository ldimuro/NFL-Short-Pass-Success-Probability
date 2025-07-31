import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve, auc


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


class BasicCNN(nn.Module):
    def __init__(self, in_channels=13):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1), 
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
    

def cross_validation(x, y, num_epochs=20, k=5):
    cross_val = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []

    all_train_losses, all_val_losses = [], []
    all_train_accs, all_val_accs = [], []
    all_roc_aucs, all_log_losses, all_pr_aucs = [], [], []

    best_acc = 0.0
    best_state = None

    for fold,(train_i, val_i) in enumerate(cross_val.split(x, y)):
        print('FOLD:', fold+1)

        # Datasets/Dataloaders
        train_data = TensorDataset(x[train_i], y[train_i])
        val_data = TensorDataset(x[val_i], y[val_i])
        train_dataloader = DataLoader(train_data, batch_size=128)    #128
        val_dataloader = DataLoader(val_data, batch_size=256)        #256

        ## Model/Loss/Optimizer
        model = BasicCNN()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4) # best so far: 1e-4
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - 0.59) / 0.59)) # handle class imbalance, since SUCCESS label represents 59% of data
        scheduler = OneCycleLR(optimizer, max_lr=3e-4, epochs=100, steps_per_epoch=len(train_dataloader)) # best so far: 3e-4

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

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

                pred = (torch.sigmoid(output) > 0.5)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

            train_losses.append(total_loss / len(train_dataloader))
            train_accs.append(correct / total)

        
            # Validation eval
            model.eval()
            val_loss = 0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for xb, yb in val_dataloader:
                    output = model(xb)
                    loss = criterion(output, yb.float())
                    val_loss += loss.item()
                    pred = (torch.sigmoid(output) > 0.5)
                    val_correct += (pred == yb).sum().item()
                    val_total += yb.size(0)

            val_losses.append(val_loss / len(val_dataloader))
            val_accs.append(val_correct / val_total)

            # Update best model across all folds
            val_acc = val_correct / val_total
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()

        scores.append(val_accs[-1])
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        # Evaluate accuracy
        # scores.append(accuracy(model, val_dataloader))

        # Compute ROC-AUC and Log Loss for this fold
        probs = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_dataloader:
                output = model(xb)
                prob = torch.sigmoid(output).squeeze().cpu().numpy()
                label = yb.squeeze().cpu().numpy()
                probs.extend(prob.tolist())
                true_labels.extend(label.tolist())

        roc = roc_auc_score(true_labels, probs)
        logloss = log_loss(true_labels, probs)

        precision, recall, _ = precision_recall_curve(true_labels, probs)
        pr_auc = auc(recall, precision)

        all_roc_aucs.append(roc)
        all_log_losses.append(logloss)
        all_pr_aucs.append(pr_auc)
        

        print(f"\tFold {fold+1} ROC-AUC: {roc:.4f}, PR-AUC: {pr_auc:.4f}, Log Loss: {logloss:.4f}")

    # Plot average over folds
    avg_train_loss = np.mean(all_train_losses, axis=0)
    avg_val_loss   = np.mean(all_val_losses, axis=0)
    avg_train_acc  = np.mean(all_train_accs, axis=0)
    avg_val_acc    = np.mean(all_val_accs, axis=0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_train_acc, label='Train Acc')
    plt.plot(avg_val_acc, label='Val Acc')
    plt.title(f"Accuracy (Best={best_acc*100:.2f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_val_loss_accuracy.png')

    print(f"\nAvg ROC-AUC across folds: {np.mean(all_roc_aucs):.4f} +/- {np.std(all_roc_aucs):.4f}")
    print(f"Avg PR-AUC across folds: {np.mean(all_pr_aucs):.4f} +/- {np.std(all_pr_aucs):.4f}")
    print(f"Avg Log Loss across folds: {np.mean(all_log_losses):.4f} +/- {np.std(all_log_losses):.4f}")


    return np.mean(scores), np.std(scores), best_acc, best_state

