import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "../data/processed/tumor_volumes.csv"
GRAPHS_DIR = "graphs"

def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    print(f"Number of patients: {df['patient'].nunique()}")
    print(f"Total number of measurements: {len(df)}")
    print(f"Min measurements: {df.groupby('patient').size().min()}")
    print(f"Max measurements: {df.groupby('patient').size().max()}")
    print(f"Mean measurements: {df.groupby('patient').size().mean():.1f}")
    print(f"\nTumor volume (cm3):")
    print(f"  Min: {df['volume_cm3'].min():.2f}")
    print(f"  Max: {df['volume_cm3'].max():.2f}")
    print(f"  Mean: {df['volume_cm3'].mean():.2f}")
    print(f"  Median: {df['volume_cm3'].median():.2f}")

    print(f"Duplicates (same week, more than one measurement):")
    duplicates = df.groupby(['patient', 'week']).size()
    duplicates = duplicates[duplicates > 1]
    print(f"Patients with duplicate measurements: {len(duplicates)}")
    print(duplicates.to_string())

    df_clean = (df.groupby(['patient', 'week'])['volume_cm3']
                  .mean()
                  .reset_index())

    patients = df_clean['patient'].unique()[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, patient in enumerate(patients):
        data = df_clean[df_clean['patient'] == patient].sort_values('week')
        axes[i].plot(data['week'], data['volume_cm3'],
                     'o-', color='steelblue', linewidth=2, markersize=6)
        axes[i].set_title(patient, fontsize=10)
        axes[i].set_xlabel('Week')
        axes[i].set_ylabel('Volume (cm3)')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Tumor growth dynamic — first 12 patients', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/tumor_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nGraph saved: {GRAPHS_DIR}/tumor_dynamics.png")

    measurements = df.groupby('patient').size()

    plt.figure(figsize=(8, 4))
    plt.hist(measurements,
             bins=range(1, measurements.max() + 2),
             color='steelblue', edgecolor='white', alpha=0.8)
    plt.xlabel('Number of measurements per patient')
    plt.ylabel('Number of patients')
    plt.title('Measurements distribution per patient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{GRAPHS_DIR}/measurements_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graph saved: {GRAPHS_DIR}/measurements_distribution.png")


if __name__ == '__main__':
    main()