import torch
import pandas as pd
import os
from tqdm import tqdm

CHUNK_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\pcqm4m-v2\pcqm4m-v2\processed_chunks"
CSV_PATH = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\pcqm4m-v2\pcqm4m-v2\raw\data.csv.gz"
OUTPUT_FILE = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\final_dataset_with_targets.pt"

# Load Targets into Memory 
print(" Loading target values from CSV...")
# Only load needed columns to save memory
target_df = pd.read_csv(CSV_PATH, compression="gzip", usecols=["smiles", "homolumogap"])
smiles_to_target = dict(zip(target_df["smiles"], target_df["homolumogap"]))
print(f" Loaded {len(smiles_to_target)} target mappings")

# Merge Chunks & Add Targets 
print(" Merging chunks and adding targets...")
combined_data = []
chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".pt")])

for file in tqdm(chunk_files, desc="Processing Chunks"):
    path = os.path.join(CHUNK_DIR, file)
    chunk = torch.load(path)
    
    for item in chunk:
        smiles = item["smiles"]
        target = smiles_to_target.get(smiles)
        
        if target is not None:
            item["target"] = float(target)  
            combined_data.append(item)
        

# Save Final Dataset 
print(f" Saving final dataset ({len(combined_data)} samples)...")
torch.save(combined_data, OUTPUT_FILE)
print(f" Success! Saved to: {OUTPUT_FILE}")

# Verify 
print("\n--- Verification ---")
verify_data = torch.load(OUTPUT_FILE)
print("Keys:", verify_data[0].keys())
print("Target Present:", 'target' in verify_data[0])
print("Sample Target:", verify_data[0].get('target'))