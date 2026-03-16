import os
import nibabel as nib
import numpy as np
import pandas as pd

DATA_DIR = "../data/train/train"
OUTPUT_DIR = "../data/processed"
OUTPUT_CSV = "../data/processed/tumor_volumes.csv"


def get_voxel_volumes(nii_path):
    """
    Svaki NIfTI fajl u header-u cuva fizicku velicinu jednog voksela u mm.
    Mnozimo x * y * z dimenzije da dobijemo zapreminu jednog voksela.
    """
    img = nib.load(nii_path)
    dims = img.header.get_zooms()[:3]
    return float(dims[0]) * float(dims[1]) * float(dims[2])

def get_tumor_volume_cm3(nii_path):
    """
    seg_mask.nii sadrzi 0 gde nema tumora i vrednosti > 0 gde ima tumora.
    Brojimo koliko voksela pripada tumoru i mnozimo sa zapreminom voksela.
    Na kraju konvertujemo mm3 u cm3.
    """
    voxel_volume = get_voxel_volumes(nii_path)

    img = nib.load(nii_path)
    mask = img.get_fdata()

    tumor_voxels = np.sum(mask > 0)

    return tumor_voxels * voxel_volume / 1000.0

def get_week_number(week_folder):
    parts = week_folder.replace("week-", "").split("-")
    return int(parts[0])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    records = []

    patient_list = sorted([
        p for p in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, p))
    ])

    print(f"Found {len(patient_list)} patients\n")

    for patient in patient_list:
        patient_path = os.path.join(DATA_DIR, patient)

        week_list = sorted([
            w for w in os.listdir(patient_path)
            if os.path.isdir(os.path.join(patient_path, w))
            and w.startswith("week-")
        ])

        for week_folder in week_list:
            seg_path = os.path.join(patient_path, week_folder, "seg_mask.nii")

            if not os.path.exists(seg_path):
                print(f"{seg_path} is missing")
                continue

            volume = get_tumor_volume_cm3(seg_path)
            week = get_week_number(week_folder)

            records.append({
                "patient": patient,
                "week_folder": week_folder,
                "week": week,
                "volume_cm3": volume,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["patient", "week", "week_folder"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()