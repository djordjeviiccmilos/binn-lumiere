import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv

from models.binn import BINN, BINNLoss


CSV_PATH = "../data/processed/tumor_volumes_clean.csv"
MODEL_PATH = "outputs/BINN_v1/model.pt"
LOG_PATH = "outputs/BINN_v1/training_log.csv"

HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 500
BIOLOGY_WEIGHT = 0.05
EARLY_STOPPING_PATIENCE = 50

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_patient_data(df, patient_id):
    """
    Za jednog pacijenta vraca:
        t        - normalizovane vremenske tacke
        V        - normalizovane zapremine
        features - karakteristike pacijenata koje ulaze u mrezu
    """
    data = df[df['patient'] == patient_id].sort_values('week_normalized')

    t = torch.tensor(data['week_normalized'].values, dtype=torch.float32)
    t = t / t.max() if t.max() > 0 else t

    V = torch.tensor(data['volume_normalized'].values, dtype=torch.float32)

    initial_volume = data['initial_volume'].iloc[0]
    n_measurements = len(data)
    max_week = data['week_normalized'].max()

    features = torch.tensor([
        1.0,
        np.log1p(initial_volume),
        n_measurements / 20.0,
        max_week / 173.0
    ], dtype=torch.float32)

    return t, V, features


def train():
    df = pd.read_csv(CSV_PATH)

    # Treniramo na svim pacijentima
    # Metrike za evaluaciju dolaze iz kfold treninga
    patients = df['patient'].unique()

    model = BINN(hidden_size=HIDDEN_SIZE).to(DEVICE)
    criterion = BINNLoss(biology_weight=BIOLOGY_WEIGHT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    log_file = open(LOG_PATH, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch",
        "train_loss", "train_data_loss", "train_biology_loss",
        "train_mae", "train_mape",
        "mean_alpha", "mean_K", "mean_beta"
    ])

    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_data_loss = train_biology_loss = 0.0
        train_mae = train_mape = 0.0

        for patient_id in patients:
            t, V, features = get_patient_data(df, patient_id)
            t = t.to(DEVICE)
            V = V.to(DEVICE)
            features = features.unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()

            V_pred, alpha, K, beta = model(features, t)
            V_pred = V_pred.squeeze()

            loss, data_loss, biology_loss = criterion(V_pred, V, alpha, K, beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_data_loss += data_loss.item()
            train_biology_loss += biology_loss.item()

            with torch.no_grad():
                pred = V_pred.detach().cpu().numpy()
                true = V.cpu().numpy()
                train_mae += np.mean(np.abs(pred - true))
                mask = true > 1e-6
                if mask.any():
                    train_mape += np.mean(
                        np.abs((pred[mask] - true[mask]) / true[mask])
                    ) * 100

        n = len(patients)
        train_loss /= n
        train_data_loss /= n
        train_biology_loss /= n
        train_mae /= n
        train_mape /= n

        scheduler.step(train_loss)

        all_alpha, all_K, all_beta = [], [], []
        with torch.no_grad():
            for patient_id in patients:
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
            f"{train_loss:.6f}", f"{train_data_loss:.6f}",
            f"{train_biology_loss:.6f}", f"{train_mae:.6f}", f"{train_mape:.4f}",
            f"{mean_alpha:.4f}", f"{mean_K:.4f}", f"{mean_beta:.4f}"
        ])
        log_file.flush()

        print(
            f"Epoch {epoch + 1:4d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.1f}% | "
            f"alpha: {mean_alpha:.3f} | K: {mean_K:.3f} | beta: {mean_beta:.3f}"
        )

        # Early stopping na osnovu train lossa
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping on epoch {epoch + 1}!")
                break

    log_file.close()
    print(f"\nBest train loss: {best_train_loss:.4f}")
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()