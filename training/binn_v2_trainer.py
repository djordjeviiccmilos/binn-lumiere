import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
import os

from models.binn_v2 import PatientBINN, BINNLossV2


CSV_PATH = "data/processed/tumor_volumes_clean.csv"
MODEL_PATH = "training/outputs/BINN_v2/model_v2.pt"
LOG_PATH = "training/outputs/BINN_v2/training_log_v2.csv"
PARAMS_PATH = "training/outputs/BINN_v2/patient_parameters_v2.csv"

LEARNING_RATE = 5e-2   # nema skrivenih slojeva - direktni parametri konvergiraju brze
EPOCHS = 500
BIOLOGY_WEIGHT = 0.01
EARLY_STOPPING_PATIENCE = 50
K_MARGIN = 1.5          # K_max po pacijentu = K_MARGIN * max opazene zapremine
K_MAX_FLOOR = 3.0        # donja granica da K_max ne bude nerealno mali kod pacijenata sa malo rasta

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_patient_data(df, patient_id):
    """
    Za jednog pacijenta vraca:
        t          - normalizovane vremenske tacke
        V          - normalizovane zapremine
        K_max_i    - gornja granica za K, izracunata iz stvarno opazene
                     zapremine OVOG pacijenta (Option E - data-driven bound)
    """
    data = df[df['patient'] == patient_id].sort_values('week_normalized')

    t = torch.tensor(data['week_normalized'].values, dtype=torch.float32)
    t = t / t.max() if t.max() > 0 else t

    V = torch.tensor(data['volume_normalized'].values, dtype=torch.float32)

    K_max_i = max(K_MARGIN * V.max().item(), K_MAX_FLOOR)

    return t, V, K_max_i


def train():
    df = pd.read_csv(CSV_PATH)
    patients = df['patient'].unique()
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    n_patients = len(patients)

    # Predracunavanje svih pacijentovih podataka i K_max granica jednom,
    # umesto u svakoj epohi (podaci se ne menjaju tokom treninga)
    patient_data = {}
    for patient_id in patients:
        t, V, K_max_i = get_patient_data(df, patient_id)
        patient_data[patient_id] = (t.to(DEVICE), V.to(DEVICE), K_max_i)

    model = PatientBINN(n_patients=n_patients).to(DEVICE)
    criterion = BINNLossV2(biology_weight=BIOLOGY_WEIGHT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PARAMS_PATH), exist_ok=True)

    log_file = open(LOG_PATH, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch",
        "train_loss", "train_data_loss", "train_biology_loss",
        "train_mae", "train_mape",
        "mean_alpha", "mean_K", "mean_beta_start", "mean_beta_end", "mean_t_switch"
    ])

    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_data_loss = train_biology_loss = 0.0
        train_mae = train_mape = 0.0

        for patient_id in patients:
            idx = patient_to_idx[patient_id]
            t, V, K_max_i = patient_data[patient_id]

            optimizer.zero_grad()

            V_pred, alpha, K, beta_start, beta_end, t_switch = model(idx, t)

            loss, data_loss, biology_loss = criterion(
                V_pred, V, alpha, K, beta_start, beta_end, K_max_override=K_max_i
            )
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

        n = n_patients
        train_loss /= n
        train_data_loss /= n
        train_biology_loss /= n
        train_mae /= n
        train_mape /= n

        scheduler.step(train_loss)

        all_alpha, all_K, all_beta_start, all_beta_end, all_t_switch = [], [], [], [], []
        with torch.no_grad():
            for patient_id in patients:
                idx = patient_to_idx[patient_id]
                alpha, K, beta_start, beta_end, t_switch = model.get_params(idx)
                all_alpha.append(alpha.item())
                all_K.append(K.item())
                all_beta_start.append(beta_start.item())
                all_beta_end.append(beta_end.item())
                all_t_switch.append(t_switch.item())

        mean_alpha = np.mean(all_alpha)
        mean_K = np.mean(all_K)
        mean_beta_start = np.mean(all_beta_start)
        mean_beta_end = np.mean(all_beta_end)
        mean_t_switch = np.mean(all_t_switch)

        log_writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}", f"{train_data_loss:.6f}",
            f"{train_biology_loss:.6f}", f"{train_mae:.6f}", f"{train_mape:.4f}",
            f"{mean_alpha:.4f}", f"{mean_K:.4f}", f"{mean_beta_start:.4f}", f"{mean_beta_end:.4f}", f"{mean_t_switch:.4f}"
        ])
        log_file.flush()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.1f}% | "
                f"alpha: {mean_alpha:.3f} | K: {mean_K:.3f} | "
                f"beta_start: {mean_beta_start:.3f} | beta_end: {mean_beta_end:.3f} | t_switch: {mean_t_switch:.3f}"
            )

        # Early stopping na osnovu train lossa
        if train_loss < best_train_loss - 5e-6:
            best_train_loss = train_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping on epoch {epoch + 1}!")
                break

    log_file.close()

    # Sacuvaj interpretabilne parametre po pacijentu - direktno koristivo
    # za klinicku diskusiju (npr. "za ovog pacijenta model procenjuje da je
    # terapija pocela da deluje na t_switch=0.42 normalizovanog vremena")
    with open(PARAMS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "alpha", "K", "beta_start", "beta_end", "t_switch"])
        with torch.no_grad():
            for patient_id in patients:
                idx = patient_to_idx[patient_id]
                alpha, K, beta_start, beta_end, t_switch = model.get_params(idx)
                writer.writerow([
                    patient_id,
                    f"{alpha.item():.4f}", f"{K.item():.4f}",
                    f"{beta_start.item():.4f}", f"{beta_end.item():.4f}", f"{t_switch.item():.4f}"
                ])

    print(f"\nBest train loss: {best_train_loss:.4f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Per-patient parameters saved: {PARAMS_PATH}")


if __name__ == "__main__":
    train()