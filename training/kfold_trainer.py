import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import csv
import os
import sys

from models.binn import BINN, BINNLoss
from binn_trainer import get_patient_data


CSV_PATH = "../data/processed/tumor_volumes_clean.csv"
OUTPUT_DIR = "outputs"

HIDDEN_SIZE = 64
LEARNING_RATE = 5e-3
EPOCHS = 500
BIOLOGY_WEIGHT = 0.01
EARLY_STOPPING_PATIENCE = 50
N_FOLDS = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_fold(df, train_patients, val_patients, fold_idx):
    """
    Trenira model na jednom foldu i vraca najbolji val loss i metrike.
    """
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"Train: {len(train_patients)} | Val: {len(val_patients)}")
    print(f"{'='*60}")

    model = BINN(hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = BINNLoss(biology_weight=BIOLOGY_WEIGHT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_mape = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    log_path = f"{OUTPUT_DIR}/fold_{fold_idx + 1}_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch",
        "train_loss", "train_mae", "train_mape",
        "val_loss", "val_mae", "val_mape",
        "mean_alpha", "mean_K", "mean_beta"
    ])

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_mae = train_mape = 0.0

        for patient_id in train_patients:
            t, V, features = get_patient_data(df, patient_id)
            t = t.to(DEVICE)
            V = V.to(DEVICE)
            features = features.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()

            V_pred, alpha, K, beta = model(features, t)
            V_pred = V_pred.squeeze()

            loss, data_loss, biology_loss = criterion(V_pred, V, alpha, K, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                pred = V_pred.detach().cpu().numpy()
                true = V.cpu().numpy()
                train_mae += np.mean(np.abs(pred - true))
                mask = true > 1e-6
                if mask.any():
                    train_mape += np.mean(
                        np.abs((pred[mask] - true[mask]) / true[mask])
                    ) * 100

        n = len(train_patients)
        train_loss /= n
        train_mae /= n
        train_mape /= n

        model.eval()
        val_loss = val_mae = val_mape = 0.0

        with torch.no_grad():
            for patient_id in val_patients:
                t, V, features = get_patient_data(df, patient_id)
                t = t.to(DEVICE)
                V = V.to(DEVICE)
                features = features.unsqueeze(0).to(DEVICE)

                V_pred, alpha, K, beta = model(features, t)
                V_pred = V_pred.squeeze()

                loss, data_loss, biology_loss = criterion(V_pred, V, alpha, K, beta)

                val_loss += loss.item()
                pred = V_pred.cpu().numpy()
                true = V.cpu().numpy()
                val_mae += np.mean(np.abs(pred - true))
                mask = true > 1e-6
                if mask.any():
                    val_mape += np.mean(
                        np.abs((pred[mask] - true[mask]) / true[mask])
                    ) * 100

        n = len(val_patients)
        val_loss /= n
        val_mae /= n
        val_mape /= n

        scheduler.step(val_loss)

        all_alpha, all_K, all_beta = [], [], []
        with torch.no_grad():
            for patient_id in train_patients:
                t, V, features = get_patient_data(df, patient_id)
                features = features.unsqueeze(0).to(DEVICE)
                t = t.to(DEVICE)
                _, alpha, K, beta = model(features, t)
                all_alpha.append(alpha.item())
                all_K.append(K.item())
                all_beta.append(beta.item())

        mean_alpha = np.mean(all_alpha)
        mean_K = np.mean(all_K)
        mean_beta = np.mean(all_beta)

        log_writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}", f"{train_mae:.6f}", f"{train_mape:.4f}",
            f"{val_loss:.6f}", f"{val_mae:.6f}", f"{val_mape:.4f}",
            f"{mean_alpha:.4f}", f"{mean_K:.4f}", f"{mean_beta:.4f}"
        ])
        log_file.flush()

        print(
            f"Epoch {epoch+1:4d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.1f}% || "
            f"Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.1f}% | "
            f"alpha: {mean_alpha:.3f} | K: {mean_K:.3f} | beta: {mean_beta:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_mape = val_mape
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                f"{OUTPUT_DIR}/best_model_fold_{fold_idx + 1}.pt"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping on epoch {epoch + 1}!")
                break

    log_file.close()

    print(f"\nFold {fold_idx + 1} done!")
    print(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Best val MAE:  {best_val_mae:.4f}")
    print(f"  Best val MAPE: {best_val_mape:.2f}%")

    return {
        "fold":      fold_idx + 1,
        "val_loss":  best_val_loss,
        "val_mae":   best_val_mae,
        "val_mape":  best_val_mape,
        "best_epoch": best_epoch
    }

def train_kfold():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    patients = df['patient'].unique()

    print(f"Number of patients: {len(patients)}")
    print(f"Number of folds: {N_FOLDS}")

    # KFold split
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patients)):
        train_patients = patients[train_idx]
        val_patients = patients[val_idx]

        result = train_fold(df, train_patients, val_patients, fold_idx)
        fold_results.append(result)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{OUTPUT_DIR}/kfold_results.csv", index=False)

    print(f"\n{'='*60}")
    print("K-fold training summary")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    print(f"\nMean val loss:  {results_df['val_loss'].mean():.4f} "
          f"± {results_df['val_loss'].std():.4f}")
    print(f"Mean val MAE:   {results_df['val_mae'].mean():.4f} "
          f"± {results_df['val_mae'].std():.4f}")
    print(f"Mean val MAPE:  {results_df['val_mape'].mean():.2f}% "
          f"± {results_df['val_mape'].std():.2f}%")

    # Cuva najbolji fold kao finalni model
    best_fold = results_df.loc[results_df['val_loss'].idxmin(), 'fold']
    print(f"\nBest fold: {best_fold}")

    import shutil
    shutil.copy(
        f"{OUTPUT_DIR}/best_model_fold_{best_fold}.pt",
        f"{OUTPUT_DIR}/best_model.pt"
    )


if __name__ == "__main__":
    train_kfold()