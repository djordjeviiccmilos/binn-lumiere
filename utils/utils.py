"""
Pomocne funkcije za vizuelizaciju i analizu BINN v2 rezultata.

Sadrzi iste grafike kao results_summary.ipynb (tok treninga, raspodela
parametara, najbolje/najgore predikcije), plus jedan nov grafik specifican
za v2 model: beta(t) kriva po pacijentu, koja pokazuje KAKO se model-ovana
efikasnost terapije menja tokom vremena za konkretnog pacijenta - ovo je
direktno koristno za "medicinsku transparentnost" diskusiju u tezi, jer
se moze doslovno pokazati zasto je model predvideo rast ili smanjenje
zapremine u odredjenom periodu.

Upotreba (u notebook-u):

    from models.binn_v2 import PatientBINN
    from training.binn_trainer_v2 import get_patient_data
    from utils import *

    model = PatientBINN(n_patients=len(patients))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    results_df = compute_results_df(model, df, patient_to_idx, DEVICE)
    plot_training_curves(LOG_PATH, save_path="results/training_curves_v2.png")
    plot_parameter_analysis(results_df, save_path="results/parameter_analysis_v2.png")
    plot_parameter_distributions(results_df, save_path="results/parameter_distributions_v2.png")
    plot_predictions(model, df, results_df, patient_to_idx, DEVICE, save_path="results/predictions_v2.png")
    plot_beta_curves(model, df, results_df, patient_to_idx, DEVICE, save_path="results/beta_curves_v2.png")
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from models.binn_v2 import PatientBINN


# --- SAMO IZMENI OVO ---
MODEL_PATH = "training/outputs/BINN_v2/model_v2.pt"
CSV_PATH = "data/processed/tumor_volumes_clean.csv"
PARAMS_PATH = "training/outputs/BINN_v2/patient_parameters_v2.csv"
LOG_PATH = "training/outputs/BINN_v2/training_log_v2.csv"
OUTPUT_DIR = "results/BINN_v2/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# -----------------------


def _tumor_grows(df, patient_id):
    data = df[df['patient'] == patient_id].sort_values('week_normalized')
    first_volume = data['volume_normalized'].iloc[0]
    last_volume = data['volume_normalized'].iloc[-1]
    return last_volume > first_volume


def _get_patient_tensors(df, patient_id, device):
    """Lokalna verzija get_patient_data (bez potrebe za importom iz trainer-a)."""
    data = df[df['patient'] == patient_id].sort_values('week_normalized')
    t = torch.tensor(data['week_normalized'].values, dtype=torch.float32)
    t = t / t.max() if t.max() > 0 else t
    V = torch.tensor(data['volume_normalized'].values, dtype=torch.float32)
    return t.to(device), V.to(device)


def compute_results_df(model, df, patient_to_idx, device="cpu"):
    """
    Prolazi kroz sve pacijente, racuna predikcije i MAE/MAPE, i vraca
    DataFrame sa svim naucenim parametrima - osnova za sve ostale plotove
    u ovom fajlu. Ekvivalent cell 6 iz results_summary.ipynb, prilagodjen
    za v2 model (beta_start/beta_end/t_switch umesto jedinstvenog beta).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for patient_id, idx in patient_to_idx.items():
            t, V = _get_patient_tensors(df, patient_id, device)

            V_pred, alpha, K, beta_start, beta_end, t_switch = model(idx, t)

            pred = V_pred.cpu().numpy()
            true = V.cpu().numpy()

            mae = np.mean(np.abs(pred - true))
            mask = true > 1e-6
            mape = np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100 if mask.any() else np.nan

            results.append({
                "patient":        patient_id,
                "alpha":          alpha.item(),
                "K":              K.item(),
                "beta_start":     beta_start.item(),
                "beta_end":       beta_end.item(),
                "t_switch":       t_switch.item(),
                "tumor_grows":    _tumor_grows(df, patient_id),
                "mae":            mae,
                "mape":           mape,
                "initial_vol":    df[df['patient'] == patient_id]['initial_volume'].iloc[0],
                "n_measurements": len(true),
            })

    return pd.DataFrame(results)


def run_mann_whitney(results_df, params=("alpha", "K", "beta_start", "beta_end")):
    """Isti Mann-Whitney U test kao u notebook-u (cell 10), sada i za
    beta_start/beta_end umesto jedinstvenog beta."""
    grows = results_df[results_df['tumor_grows'] == True]
    shrinks = results_df[results_df['tumor_grows'] == False]

    for param in params:
        stat, p = stats.mannwhitneyu(grows[param].values, shrinks[param].values, alternative='two-sided')
        print(f"{param:12s} | p = {p:.6f} | {'znacajno' if p < 0.05 else 'nije znacajno'}")


def plot_training_curves(log_path, save_path=None, show=True):
    """Loss / MAE / MAPE kroz epohe. Isto kao notebook cell 12, radi i sa
    v1 i v2 log fajlom (kolone koje ne postoje se preskacu)."""
    log = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(log['epoch'], log['train_loss'], color='steelblue')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoha')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log['epoch'], log['train_mae'], color='steelblue')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoha')
    axes[1].set_ylabel('MAE')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(log['epoch'], log['train_mape'], color='steelblue')
    axes[2].set_title('MAPE')
    axes[2].set_xlabel('Epoha')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_parameter_analysis(results_df, save_path=None, show=True):
    """Boxplot-ovi rastuci vs. smanjujuci tumori. Isto kao notebook cell 15,
    prosireno na 4 panela (beta_start i beta_end umesto jednog beta)."""
    grows = results_df[results_df['tumor_grows'] == True]
    shrinks = results_df[results_df['tumor_grows'] == False]

    params = [
        ('alpha',      'Alpha (brzina rasta)'),
        ('K',          'K (maksimalna zapremina)'),
        ('beta_start', 'Beta na pocetku praćenja'),
        ('beta_end',   'Beta na kraju praćenja'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    for i, (param, label) in enumerate(params):
        axes[i].boxplot(
            [grows[param].values, shrinks[param].values],
            tick_labels=['Raste', 'Smanjuje se']
        )
        axes[i].set_title(param.upper())
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Bioloski parametri rastuci vs. smanjujuci tumori', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_parameter_distributions(results_df, save_path=None, show=True):
    """Histogrami raspodele parametara po svim pacijentima. Isto kao
    notebook cell 20, prosireno na 5 panela (dodat t_switch)."""
    params = [
        ('alpha',      'steelblue'),
        ('K',          'seagreen'),
        ('beta_start', 'tomato'),
        ('beta_end',   'orchid'),
        ('t_switch',   'goldenrod'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for ax, (param, color) in zip(axes, params):
        ax.hist(results_df[param], bins=20, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(results_df[param].mean(), color='black', linestyle='--',
                   label=f'Prosecna vrednost: {results_df[param].mean():.3f}')
        ax.axvline(results_df[param].median(), color='orange', linestyle='-',
                   label=f'Medijalna vrednost: {results_df[param].median():.3f}')
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.set_ylabel('Broj pacijenata')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Raspodela parametara', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_predictions(model, df, results_df, patient_to_idx, device="cpu",
                      n_best=6, n_worst=6, save_path=None, show=True):
    """3 najbolje i 3 najgore predikcije rangirane po MAE. Isto kao
    notebook cell 18, prilagodjeno v2 modelu (poziv model(idx, t))."""
    model.eval()

    best_patients = results_df.nsmallest(n_best, 'mae')['patient'].tolist()
    worst_patients = results_df.nlargest(n_worst, 'mae')['patient'].tolist()

    n_cols = max(n_best, n_worst)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    axes = np.array(axes).reshape(2, n_cols)

    with torch.no_grad():
        for row, patient_list in enumerate([best_patients, worst_patients]):
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(patient_list):
                    ax.axis('off')
                    continue

                patient_id = patient_list[col]
                idx = patient_to_idx[patient_id]
                t, V = _get_patient_tensors(df, patient_id, device)

                V_pred, alpha, K, beta_start, beta_end, t_switch = model(idx, t)
                V_pred = V_pred.cpu().numpy()
                V_true = V.cpu().numpy()
                t_np = t.cpu().numpy()

                r = results_df[results_df['patient'] == patient_id].iloc[0]
                status = "raste" if r['tumor_grows'] else "smanjuje se"

                ax.plot(t_np, V_true, 'o-', label='Stvarna vrednost', color='steelblue', linewidth=2)
                ax.plot(t_np, V_pred, 's--', label='Predikcija', color='tomato', linewidth=2)
                ax.set_title(
                    f"{patient_id}\n"
                    f"alpha={alpha.item():.2f} K={K.item():.2f} "
                    f"beta={beta_start.item():.2f}->{beta_end.item():.2f} ({status})",
                    fontsize=9
                )
                ax.set_xlabel('Vreme')
                ax.set_ylabel('Zapremina')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.55)

    # Naslovi redova se postavljaju NAKON tight_layout/subplots_adjust, na
    # osnovu stvarne (finalne) pozicije prvog subplota u svakom redu -
    # fiksne figure-relativne koordinate (npr. y=0.5) se ne poklapaju
    # pouzdano sa drugim redom jer zavise od broja kolona i figsize.
    row_titles = ['Najbolje predikcije', 'Najgore predikcije']
    for row_idx, title in enumerate(row_titles):
        bbox = axes[row_idx, 0].get_position()
        fig.text(0.5, bbox.y1 + 0.02, title, ha='center', fontsize=12, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_beta_curves(model, df, results_df, patient_to_idx, device="cpu",
                      patient_ids=None, n_patients=6, save_path=None, show=True):
    """
    NOVO u v2: prikazuje naucenu beta(t) krivu (efekat terapije kroz vreme)
    za odabrane pacijente, zajedno sa stvarnom zapreminom na istoj vremenskoj
    osi. Ovo je kljucan grafik za "medicinsku transparentnost" - direktno
    pokazuje KADA i U KOM SMERU model procenjuje da se efekat terapije
    promenio, umesto da to bude skriveno unutar jednog broja.

    Ako patient_ids nije prosledjen, bira n_patients pacijenata sa najvecom
    razlikom |beta_start - beta_end| (najvise "interesantni" slucajevi gde
    se efekat terapije najvise promenio tokom praćenja).
    """
    model.eval()

    if patient_ids is None:
        results_df = results_df.copy()
        results_df['beta_shift'] = (results_df['beta_start'] - results_df['beta_end']).abs()
        patient_ids = results_df.nlargest(n_patients, 'beta_shift')['patient'].tolist()

    n = len(patient_ids)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for i, patient_id in enumerate(patient_ids):
            ax = axes[i]
            idx = patient_to_idx[patient_id]
            t, V = _get_patient_tensors(df, patient_id, device)

            t_dense = torch.linspace(0, 1, 100, device=device)
            beta_dense = model.beta_curve(idx, t_dense).cpu().numpy()

            V_pred, alpha, K, beta_start, beta_end, t_switch = model(idx, t)

            ax2 = ax.twinx()
            ax.plot(t.cpu().numpy(), V.cpu().numpy(), 'o-', color='steelblue',
                    label='Stvarna zapremina', linewidth=2)
            ax.plot(t.cpu().numpy(), V_pred.cpu().numpy(), 's--', color='tomato',
                    label='Predikcija', linewidth=2, alpha=0.8)
            ax2.plot(t_dense.cpu().numpy(), beta_dense, color='seagreen',
                     label='beta(t)', linewidth=2, alpha=0.7)
            ax2.axvline(t_switch.item(), color='gray', linestyle=':', alpha=0.7)

            ax.set_xlabel('Vreme')
            ax.set_ylabel('Zapremina', color='steelblue')
            ax2.set_ylabel('beta(t) - efekat terapije', color='seagreen')
            ax.set_title(
                f"{patient_id}  (t_switch={t_switch.item():.2f})\n"
                f"beta: {beta_start.item():.2f} -> {beta_end.item():.2f}",
                fontsize=9
            )
            ax.grid(True, alpha=0.3)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')

        for j in range(n, len(axes)):
            axes[j].axis('off')

    plt.suptitle('Naucena dinamika efekta terapije [beta(t)] po pacijentu', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)

    # patient_to_idx se gradi iz PARAMS_PATH (patient_parameters_v2.csv), NE iz
    # sirovog CSV-a - jer je taj fajl pisan u binn_v2_trainer.py sa
    # `for patient_id in patients:` istim redosledom kojim je model treniran,
    # pa red u tom fajlu = tacan idx koji ocekuje model. Ako bismo redosled
    # ponovo izvodili iz CSV_PATH (npr. df['patient'].unique()), rizikujemo
    # da se parametri neprimetno pripisu pogresnim pacijentima ako se
    # redosled makar malo razlikuje (sortiranje, filtrirani redovi, itd.)
    params_df = pd.read_csv(PARAMS_PATH)
    patients_ordered = params_df['patient'].tolist()
    patient_to_idx = {p: i for i, p in enumerate(patients_ordered)}
    n_patients = len(patients_ordered)
    print(f"Ucitano {n_patients} pacijenata iz {PARAMS_PATH} (redosled = trening idx)")

    model = PatientBINN(n_patients=n_patients).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model ucitan iz {MODEL_PATH}")

    df = pd.read_csv(CSV_PATH)

    results_df = compute_results_df(model, df, patient_to_idx, device)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results_df.csv"), index=False)
    print("Sacuvano: results_df.csv")

    # with open(os.path.join(OUTPUT_DIR, "mann_whitney_results.txt"), "w") as f:
    #     old_stdout = sys.stdout
    #     sys.stdout = f
    #     try:
    #         run_mann_whitney(results_df)
    #     finally:
    #         sys.stdout = old_stdout
    # print("Sacuvano: mann_whitney_results.txt")

    # plot_training_curves(
    #     LOG_PATH, save_path=os.path.join(OUTPUT_DIR, "training_curves_v2.png"), show=False
    # )
    # print("Sacuvano: training_curves_v2.png")

    # plot_parameter_analysis(
    #     results_df, save_path=os.path.join(OUTPUT_DIR, "parameter_analysis_v2.png"), show=False
    # )
    # print("Sacuvano: parameter_analysis_v2.png")

    # plot_parameter_distributions(
    #     results_df, save_path=os.path.join(OUTPUT_DIR, "parameter_distributions_v2.png"), show=False
    # )
    # print("Sacuvano: parameter_distributions_v2.png")

    plot_predictions(
        model, df, results_df, patient_to_idx, device,
        save_path=os.path.join(OUTPUT_DIR, "predictions_v2.png"), show=False
    )
    print("Sacuvano: predictions_v2.png")

    # plot_beta_curves(
    #     model, df, results_df, patient_to_idx, device,
    #     save_path=os.path.join(OUTPUT_DIR, "beta_curves_v2.png"), show=False
    # )
    # print("Sacuvano: beta_curves_v2.png")

    # print(f"\nSvi rezultati sacuvani u: {OUTPUT_DIR}")