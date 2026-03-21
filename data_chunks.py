import os
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

print("Torch version:", torch.__version__)

# Correct ROOT (FOLDER, not file)
ROOT = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\pcqm4m-v2\pcqm4m-v2"
CSV_PATH = os.path.join(ROOT, "raw", "data.csv.gz")

# Output paths
SAVE_DIR = os.path.join(ROOT, "processed_chunks")
CHECKPOINT_FILE = os.path.join(ROOT, "checkpoint.txt")
os.makedirs(SAVE_DIR, exist_ok=True)

# Memory-safe read
CHUNK_READ_SIZE = 10000
CHUNK_SIZE = 5000

# Resume
start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        start_idx = int(f.read().strip())
    print(f" Resuming from index: {start_idx}")

current_index = 0
save_counter = 0
data_list = []

#  MAIN LOOP
for chunk_df in pd.read_csv(CSV_PATH, compression='gzip', chunksize=CHUNK_READ_SIZE):
    for _, row in chunk_df.iterrows():
        if current_index < start_idx:
            current_index += 1
            continue

        smiles = row["smiles"]
        
        #  NEW: Skip if target is missing/NaN
        if pd.isna(row["homolumogap"]):
            current_index += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            current_index += 1
            continue

        # Include target variable
        data = {
            "smiles": smiles,
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "target": float(row["homolumogap"])  
        }

        data_list.append(data)

        #  SAVE CHUNK
        if len(data_list) >= CHUNK_SIZE:
            file_path = os.path.join(SAVE_DIR, f"part_{save_counter}.pt")
            torch.save(data_list, file_path)
            print(f" Saved chunk {save_counter} at index {current_index}")
            data_list = []
            save_counter += 1

            # Checkpoint
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(current_index))

        current_index += 1


if len(data_list) > 0:
    file_path = os.path.join(SAVE_DIR, f"part_final.pt")
    torch.save(data_list, file_path)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(current_index))

print(" Processing complete with targets!")
