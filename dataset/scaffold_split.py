"""
=============================================================================
Scaffold Split + Memory-Safe PyG File Creation
=============================================================================

This script:
1. Loads final_dataset_with_targets.pt for scaffold extraction
2. Creates scaffold-based train/val/test split using Murcko scaffolds
3. Processes PyG chunks ONE AT A TIME (memory-safe)
4. Saves separate files: train.pt, val.pt, test.pt (PyG format)

MEMORY-SAFE FEATURES:
- Processes one chunk at a time (not all in RAM)
- Saves incrementally to disk every 50k graphs
- Can optionally skip val/test to save more memory

=============================================================================
"""

import torch
import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION

# Paths
INPUT_FILE = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\final_dataset_with_targets.pt"
CHUNK_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\temp_preprocessed"
OUTPUT_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\scaffold_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Memory-safe settings
SAVE_INTERVAL = 50000  # Save to disk every 50k graphs
SKIP_VAL_TEST = True  # Set True to ONLY save training data (saves memory)

# SCAFFOLD EXTRACTION

def get_murcko_scaffold(smiles):
    """Extract Murcko scaffold from SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        return Chem.MolToSmiles(scaffold)
    except:
        return None

# HELPER: Save data incrementally to avoid RAM crash

def save_incrementally(data_list, output_path, temp_suffix=""):
    """
    Save data incrementally to avoid loading everything in RAM
    
    If temp file exists, loads it, extends with new data, and re-saves
    """
    temp_path = output_path + ".tmp"
    
    if os.path.exists(temp_path):
        # Load existing data
        existing_data = torch.load(temp_path, weights_only=False)
        existing_data.extend(data_list)
        torch.save(existing_data, temp_path)
        del existing_data
    else:
        # First save
        torch.save(data_list, temp_path)
    
    print(f"  Saved {len(data_list):,} graphs to {temp_path}")
    return temp_path

def finalize_save(temp_path, final_path):
    """Rename temp file to final file"""
    if os.path.exists(temp_path):
        os.rename(temp_path, final_path)
        print(f"  Finalized: {final_path}")

# MAIN FUNCTION (MEMORY-SAFE)

def create_scaffold_split_and_save():
    """
    Create scaffold split with memory-safe chunk processing
    """
    
    print("CREATING SCAFFOLD SPLIT (MEMORY-SAFE)")
    
    print(f"Input (dict): {INPUT_FILE}")
    print(f"Input (PyG chunks): {CHUNK_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Save interval: {SAVE_INTERVAL:,} graphs")
    print(f"Skip val/test: {SKIP_VAL_TEST}")
    
    
    # PART 1: Create scaffold split indices from original dict data
    print(f"\n[1/4] Loading original data for scaffold extraction...")
    raw_data = torch.load(INPUT_FILE, weights_only=False)
    print(f"Loaded {len(raw_data):,} samples")
    
    # Filter valid samples
    print("Filtering valid samples...")
    valid_samples = []
    for sample in raw_data:
        target = sample.get('target')
        smiles = sample.get('smiles')
        if target is not None and not pd.isna(target) and 0 < target <= 10:
            if smiles is not None:
                valid_samples.append(sample)
    
    print(f"Valid samples: {len(valid_samples):,}")
    
    # Extract scaffolds
    print("\n[2/4] Extracting Murcko scaffolds...")
    scaffold_groups = defaultdict(list)
    
    for idx, sample in enumerate(tqdm(valid_samples, desc="Scaffolds")):
        scaffold = get_murcko_scaffold(sample['smiles'])
        if scaffold is not None:
            scaffold_groups[scaffold].append(idx)
    
    print(f"Unique scaffolds: {len(scaffold_groups)}")
    
    # Split scaffolds
    print("\n[3/4] Splitting scaffolds (80/10/10)...")
    scaffold_list = list(scaffold_groups.keys())
    np.random.seed(42)
    np.random.shuffle(scaffold_list)
    
    n_train = int(TRAIN_RATIO * len(scaffold_list))
    n_val = int(VAL_RATIO * len(scaffold_list))
    
    train_scaffolds = set(scaffold_list[:n_train])
    val_scaffolds = set(scaffold_list[n_train:n_train + n_val])
    test_scaffolds = set(scaffold_list[n_train + n_val:])
    
    # Create sets of SMILES for each split
    train_smiles = set()
    val_smiles = set()
    test_smiles = set()
    
    for scaffold, indices in scaffold_groups.items():
        if scaffold in train_scaffolds:
            for idx in indices:
                train_smiles.add(valid_samples[idx]['smiles'])
        elif scaffold in val_scaffolds:
            for idx in indices:
                val_smiles.add(valid_samples[idx]['smiles'])
        else:
            for idx in indices:
                test_smiles.add(valid_samples[idx]['smiles'])
    
    print(f"Train SMILES: {len(train_smiles):,}")
    print(f"Val SMILES: {len(val_smiles):,}")
    print(f"Test SMILES: {len(test_smiles):,}")
    
    # Save indices for reference
    np.save(os.path.join(OUTPUT_DIR, 'train_idx.npy'), 
            np.array([i for i, s in enumerate(valid_samples) if s['smiles'] in train_smiles]))
    np.save(os.path.join(OUTPUT_DIR, 'val_idx.npy'), 
            np.array([i for i, s in enumerate(valid_samples) if s['smiles'] in val_smiles]))
    np.save(os.path.join(OUTPUT_DIR, 'test_idx.npy'), 
            np.array([i for i, s in enumerate(valid_samples) if s['smiles'] in test_smiles]))
    
    # PART 2: Filter PyG chunks ONE AT A TIME (MEMORY-SAFE)
    print(f"\n[4/4] Filtering PyG chunks (memory-safe)...")
    
    # Get chunk files
    chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) 
                        if f.endswith('.pt') and f != 'chunk_list.txt'])
    print(f"Found {len(chunk_files)} PyG chunk files")
    
    # Initialize counters
    train_count = 0
    val_count = 0
    test_count = 0
    
    # Temporary storage (cleared after each save)
    train_buffer = []
    val_buffer = []
    test_buffer = []
    
    # Define output paths
    train_path = os.path.join(OUTPUT_DIR, 'train.pt')
    val_path = os.path.join(OUTPUT_DIR, 'val.pt')
    test_path = os.path.join(OUTPUT_DIR, 'test.pt')
    
    # Process each chunk ONE AT A TIME
    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
        chunk_path = os.path.join(CHUNK_DIR, chunk_file)
        if not os.path.exists(chunk_path):
            print(f"Warning: {chunk_path} not found")
            continue
        
        # Load ONE chunk (not all)
        chunk = torch.load(chunk_path, weights_only=False)
        
        # Filter each graph in this chunk
        for graph in chunk:
            smiles = getattr(graph, 'smiles', None)
            if smiles is None:
                continue
            
            # Route to appropriate split
            if smiles in train_smiles:
                train_buffer.append(graph)
                train_count += 1
            elif smiles in val_smiles and not SKIP_VAL_TEST:
                val_buffer.append(graph)
                val_count += 1
            elif smiles in test_smiles and not SKIP_VAL_TEST:
                test_buffer.append(graph)
                test_count += 1
        
        # Delete chunk immediately to free memory
        del chunk
        
        # Save incrementally if buffer is large enough
        if len(train_buffer) >= SAVE_INTERVAL:
            save_incrementally(train_buffer, train_path)
            train_buffer = []  # Clear buffer
        
        if not SKIP_VAL_TEST and len(val_buffer) >= SAVE_INTERVAL:
            save_incrementally(val_buffer, val_path)
            val_buffer = []
        
        if not SKIP_VAL_TEST and len(test_buffer) >= SAVE_INTERVAL:
            save_incrementally(test_buffer, test_path)
            test_buffer = []
        
        # Optional: Print progress
        if train_count % 500000 == 0:
            print(f"  Progress: Train={train_count:,}, Val={val_count:,}, Test={test_count:,}")
    
    # Save remaining data in buffers
    if len(train_buffer) > 0:
        save_incrementally(train_buffer, train_path)
    if not SKIP_VAL_TEST and len(val_buffer) > 0:
        save_incrementally(val_buffer, val_path)
    if not SKIP_VAL_TEST and len(test_buffer) > 0:
        save_incrementally(test_buffer, test_path)
    
    # Finalize (rename .tmp files to final files)
    print(f"\nFinalizing split files...")
    finalize_save(train_path + ".tmp", train_path)
    if not SKIP_VAL_TEST:
        finalize_save(val_path + ".tmp", val_path)
        finalize_save(test_path + ".tmp", test_path)
    
    # Print summary
    
    print("SPLIT CREATION COMPLETE - SUMMARY")
    
    print(f"Train: {train_count:,} graphs → {train_path}")
    if not SKIP_VAL_TEST:
        print(f"Val:   {val_count:,} graphs → {val_path}")
        print(f"Test:  {test_count:,} graphs → {test_path}")
    else:
        print(f"Val:   Skipped (SKIP_VAL_TEST=True)")
        print(f"Test:  Skipped (SKIP_VAL_TEST=True)")
    
    total = train_count + val_count + test_count
    print(f"\nTotal graphs saved: {total:,}")
    if os.path.exists(train_path):
        print(f"File sizes:")
        print(f"  train.pt: {os.path.getsize(train_path) / (1024**3):.2f} GB")
        if not SKIP_VAL_TEST and os.path.exists(val_path):
            print(f"  val.pt:   {os.path.getsize(val_path) / (1024**3):.2f} GB")
            print(f"  test.pt:  {os.path.getsize(test_path) / (1024**3):.2f} GB")
    
    
    # Verify sample
    if train_count > 0:
        # Load just first sample for verification
        verify_data = torch.load(train_path, weights_only=False)
        if len(verify_data) > 0:
            s = verify_data[0]
            print(f"\n Sample train graph: x={s.x.shape}, y={s.y.item():.4f}")
            del verify_data
    
    return train_count, val_count, test_count

# ENTRY POINT

if __name__ == "__main__":
    train_count, val_count, test_count = create_scaffold_split_and_save()
    
    print(f"\n Next steps:")
    print(f"   1. Use {OUTPUT_DIR}/train.pt for training")
    if not SKIP_VAL_TEST:
        print(f"   2. Use {OUTPUT_DIR}/val.pt for validation")
        print(f"   3. Use {OUTPUT_DIR}/test.pt for testing")
    else:
        print(f"   2. Val/Test skipped - run again with SKIP_VAL_TEST=False when needed")
    print(f"   3. Train your model on train.pt!")