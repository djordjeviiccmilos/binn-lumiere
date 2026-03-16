import pandas as pd


CSV_PATH = "../data/processed/tumor_volumes.csv"
OUTPUT_PATH = "../data/processed/tumor_volumes_clean.csv"

"""
Pacijenti sa manje od 4 merenja se iskljucuju.
Gompertz kriva ima 3 parametara, pa je minimalni broj potrebnih tacaka 4.
"""
MIN_MEASUREMENTS = 4


def main():
    df = pd.read_csv(CSV_PATH)

    print(f"Before cleaning: {df['patient'].nunique()} patients, {len(df)} measurements")

    """
    Ciscenje duplikata
    Ako pacijent ima week-000-1 i week-000-2, uzimamo njihov prosek
    """
    df = (df.groupby(['patient', 'week'])['volume_cm3']
            .mean()
            .reset_index())

    print(f"After cleaning duplicates: {len(df)} measurements")

    """
    Iskljucujemo pacijente sa premalo merenja
    """
    measurements_per_patient = df.groupby('patient').size()
    included = measurements_per_patient[measurements_per_patient >= MIN_MEASUREMENTS].index
    excluded = measurements_per_patient[measurements_per_patient < MIN_MEASUREMENTS]

    print(f"\nExcluded patients (less than {MIN_MEASUREMENTS} measurements):")
    print(excluded.to_string())

    df = df[df['patient'].isin(included)].reset_index(drop=True)

    """
    Normalizacija vremena
    Svaki pacijent pocinje od nedelje 0
    Ako nema tacno nedelju 0, uzimamo njegovo prvo merenje kao pocetak
    """
    df['week_normalized'] = df.groupby('patient')['week'].transform(
        lambda x: x - x.min()
    )

    """
    Normalizacija zapremine.
    Svako merenje deljeno sa pocetnom zapreminom tog pacijenta.
    Tako V(0) = 1.0 za svakog pacijenta.
    V(t) > 1.0 => tumor se povecao
    V(t) < 1.0 => tumor se smanjio
    """
    initial_volume = ((df.groupby('patient'))
            .apply(
                lambda x: x.loc[x['week_normalized'].idxmin(),'volume_cm3'],include_groups=False
            )
            .rename('initial_volume'))

    df = df.merge(initial_volume, on='patient')
    df['volume_normalized'] = df['volume_cm3'] / df['initial_volume']

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAfter cleaning: {df['patient'].nunique()} patients, {len(df)} measurements")

if __name__ == "__main__":
    main()
