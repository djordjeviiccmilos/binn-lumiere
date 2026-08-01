import torch
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import csv
import os

from models.binn_v2 import PatientBINN, BINNLossV2
from training.binn_v2_trainer import get_patient_data


CSV_PATH = "data/processed/tumor_volumes_clean.csv"
OUTPUT_DIR = "training/outputs/BINN_v2/kfold_training"

LEARNING_RATE = 5e-2   # isto kao u binn_v2_trainer.py - direktni parametri, bez skrivenih slojeva
EPOCHS = 500
BIOLOGY_WEIGHT = 0.01
EARLY_STOPPING_PATIENCE = 50
N_FOLDS = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NAPOMENA O DIZAJNU
# -------------------
# PatientBINN uci parametre (alpha, K, beta_start, beta_end, t_switch) direktno
# po indeksu pacijenta - to je per-pacijent inverzni problem (ista paradigma
# kao PINN parameter-identification radovi za rast tumora), ne prediktivni
# model koji generalizuje na nepoznatog pacijenta iz featura. Zbog toga ovaj
# k-fold NE meri generalizaciju na nove pacijente (za to bi trebalo ili
# vremenski split opservacija unutar pacijenta, ili populaciona/NLME
# struktura sa fiksnim + random efektima).
#
# Ovaj k-fold umesto toga testira STABILNOST procene bioloskih parametara:
# K nezavisnih modela se treniraju do konvergencije na K disjunktnih
# podskupova pacijenata (standardna KFold particija). Ako su srednje
# vrednosti alpha/K/beta_start/beta_end/t_switch i kvalitet fita (loss/MAE/
# MAPE) konzistentni izmedju foldova, to je argument da je procena parametara
# robustna na izbor pacijenata, a ne artefakt konkretnog uzorka. Ovo je
# direktno poredivo izmedju BINNv1 i BINNv2 (manja varijansa izmedju foldova
# = stabilnija/interpretabilnija procena).


def train_fold(df, fold_patients, fold_idx):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS} | n_patients = {len(fold_patients)}")
    print(f"{'='*60}")

    patient_to_idx = {p: i for i, p in enumerate(fold_patients)}
    n_patients = len(fold_patients)

    patient_data = {}
    for patient_id in fold_patients:
        t, V, K_max_i = get_patient_data(df, patient_id)
        patient_data[patient_id] = (t.to(DEVICE), V.to(DEVICE), K_max_i)

    model = PatientBINN(n_patients=n_patients).to(DEVICE)
    criterion = BINNLossV2(biology_weight=BIOLOGY_WEIGHT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    fold_dir = f"{OUTPUT_DIR}/fold_{fold_idx + 1}"
    os.makedirs(fold_dir, exist_ok=True)

    log_path = f"{fold_dir}/training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch",
        "train_loss", "train_data_loss", "train_biology_loss",
        "train_mae", "train_mape",
        "mean_alpha", "mean_K", "mean_beta_start", "mean_beta_end", "mean_t_switch"
    ])

    best_train_loss = float("inf")
    best_train_mae = float("inf")
    best_train_mape = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = train_data_loss = train_biology_loss = 0.0
        train_mae = train_mape = 0.0

        for patient_id in fold_patients:
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

        train_loss /= n_patients
        train_data_loss /= n_patients
        train_biology_loss /= n_patients
        train_mae /= n_patients
        train_mape /= n_patients

        scheduler.step(train_loss)

        all_alpha, all_K, all_beta_start, all_beta_end, all_t_switch = [], [], [], [], []
        with torch.no_grad():
            for patient_id in fold_patients:
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
            f"{train_loss:.6f}", f"{train_data_loss:.6f}", f"{train_biology_loss:.6f}",
            f"{train_mae:.6f}", f"{train_mape:.4f}",
            f"{mean_alpha:.4f}", f"{mean_K:.4f}",
            f"{mean_beta_start:.4f}", f"{mean_beta_end:.4f}", f"{mean_t_switch:.4f}"
        ])
        log_file.flush()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.1f}% | "
                f"alpha: {mean_alpha:.3f} | K: {mean_K:.3f} | "
                f"beta_start: {mean_beta_start:.3f} | beta_end: {mean_beta_end:.3f} | t_switch: {mean_t_switch:.3f}"
            )

        if train_loss < best_train_loss - 5e-6:
            best_train_loss = train_loss
            best_train_mae = train_mae
            best_train_mape = train_mape
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{fold_dir}/best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping on epoch {epoch + 1}!")
                break

    log_file.close()

    # Ucitaj najbolji checkpoint da izvuces finalne (a ne poslednje-epoch) parametre
    model.load_state_dict(torch.load(f"{fold_dir}/best_model.pt", map_location=DEVICE))
    model.eval()

    final_alpha, final_K, final_beta_start, final_beta_end, final_t_switch = [], [], [], [], []
    with torch.no_grad():
        for patient_id in fold_patients:
            idx = patient_to_idx[patient_id]
            alpha, K, beta_start, beta_end, t_switch = model.get_params(idx)
            final_alpha.append(alpha.item())
            final_K.append(K.item())
            final_beta_start.append(beta_start.item())
            final_beta_end.append(beta_end.item())
            final_t_switch.append(t_switch.item())

    # Sacuvaj per-patient parametre za ovaj fold (korisno za dalju inspekciju)
    with open(f"{fold_dir}/patient_parameters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "alpha", "K", "beta_start", "beta_end", "t_switch"])
        for i, patient_id in enumerate(fold_patients):
            writer.writerow([
                patient_id,
                f"{final_alpha[i]:.4f}", f"{final_K[i]:.4f}",
                f"{final_beta_start[i]:.4f}", f"{final_beta_end[i]:.4f}", f"{final_t_switch[i]:.4f}"
            ])

    print(f"\nFold {fold_idx + 1} done!")
    print(f"  Best train loss: {best_train_loss:.4f} (epoch {best_epoch})")
    print(f"  Best train MAE:  {best_train_mae:.4f}")
    print(f"  Best train MAPE: {best_train_mape:.2f}%")

    return {
        "fold": fold_idx + 1,
        "n_patients": n_patients,
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "train_mae": best_train_mae,
        "train_mape": best_train_mape,
        "mean_alpha": np.mean(final_alpha),
        "mean_K": np.mean(final_K),
        "mean_beta_start": np.mean(final_beta_start),
        "mean_beta_end": np.mean(final_beta_end),
        "mean_t_switch": np.mean(final_t_switch),
        "std_alpha": np.std(final_alpha),
        "std_K": np.std(final_K),
        "std_beta_start": np.std(final_beta_start),
        "std_beta_end": np.std(final_beta_end),
        "std_t_switch": np.std(final_t_switch),
    }


def train_kfold():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    patients = df['patient'].unique()

    print(f"Number of patients: {len(patients)}")
    print(f"Number of folds: {N_FOLDS}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    # Svaki fold trenira nezavisan model na SVOM disjunktnom podskupu pacijenata
    # (koristimo test-indekse KFold-a kao particiju, ne kao held-out validaciju)
    for fold_idx, (_, fold_idx_array) in enumerate(kf.split(patients)):
        fold_patients = patients[fold_idx_array]
        result = train_fold(df, fold_patients, fold_idx)
        fold_results.append(result)

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(f"{OUTPUT_DIR}/kfold_stability_results.csv", index=False)

    print(f"\n{'='*60}")
    print("K-fold stabilnost - rezime")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    print(f"\n--- Kvalitet fita (konzistentnost izmedju foldova) ---")
    print(f"Train loss: {results_df['train_loss'].mean():.4f} \u00b1 {results_df['train_loss'].std():.4f}")
    print(f"Train MAE:  {results_df['train_mae'].mean():.4f} \u00b1 {results_df['train_mae'].std():.4f}")
    print(f"Train MAPE: {results_df['train_mape'].mean():.2f}% \u00b1 {results_df['train_mape'].std():.2f}%")

    print(f"\n--- Stabilnost bioloskih parametara (izmedju foldova) ---")
    for param in ["mean_alpha", "mean_K", "mean_beta_start", "mean_beta_end", "mean_t_switch"]:
        vals = results_df[param]
        cv = vals.std() / abs(vals.mean()) * 100 if vals.mean() != 0 else float("nan")
        print(f"{param:>18s}: {vals.mean():.4f} \u00b1 {vals.std():.4f}  (CV = {cv:.1f}%)")

    print(
        "\nNapomena: 'CV' ovde je koeficijent varijacije SREDNJE vrednosti parametra "
        "izmedju foldova - meri koliko procena parametra zavisi od toga koji su "
        "pacijenti bili u trening skupu. Ovo je stabilnost/robusnost procene, "
        "NIJE mera prediktivne generalizacije na nove pacijente."
    )


if __name__ == "__main__":
    train_kfold()