"""
=============================================================================
Preprocessing - SMILES → PyTorch Geometric Graphs (Memory-Safe, Split-Aware)
=============================================================================

Requirements:
- Read split assignments from split.py output
- Process chunks one at a time (streaming)
- Convert SMILES → PyG Data objects with OGB-style features
- Save output as chunked files: train_chunk_0.pt, val_chunk_0.pt, etc.
- NEVER load full dataset into memory

Features:
- 9D atom features (OGB PCQM4Mv2 standard)
- 3D/4D bond features (optional distance encoding)
- Target normalization: log1p + standardization (computed from sample)
- Invalid SMILES handling with logging

Output:
- preprocessed/train_chunk_0.pt, train_chunk_1.pt, ...
- preprocessed/val_chunk_0.pt, ...
- preprocessed/test_chunk_0.pt, ...
- preprocessed/normalization_stats.json

=============================================================================
"""

"""
=============================================================================
Preprocessing - SMILES → PyTorch Geometric Graphs (Memory-Safe, Split-Aware)
=============================================================================
[... full file content with import json added at top ...]
"""

import torch
import numpy as np
import pandas as pd
import os
import gc
import json  
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from config import (
    INPUT_FILE, SPLIT_DIR, PREPROCESSED_DIR, METADATA_FILE,  
    CHUNK_SIZE, USE_3D_COORDS, USE_POSITIONAL_ENCODING,
    TARGET_LOG_TRANSFORM, NORMALIZATION_SAMPLE_SIZE,
    ATOM_FEAT_DIM, BOND_FEAT_DIM, GC_INTERVAL
)



# FEATURE EXTRACTION FUNCTIONS

def get_atom_features(atom):
    """Extract 9-dimensional atom features (OGB PCQM4Mv2 standard)"""
    hybridization_dict = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    chirality_dict = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.rdchem.ChiralType.CHI_OTHER: 3,
    }
    
    return np.array([
        atom.GetAtomicNum(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        hybridization_dict.get(atom.GetHybridization(), 5),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        chirality_dict.get(atom.GetChiralTag(), 0),
        atom.GetDegree()
    ], dtype=np.float32)


def get_bond_features(bond):
    """Extract 3-dimensional bond features (OGB PCQM4Mv2 standard)"""
    bond_type_dict = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    
    return np.array([
        bond_type_dict.get(bond.GetBondType(), 4),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing())
    ], dtype=np.float32)


def generate_3d_coordinates(mol):
    """Generate 3D coordinates using ETKDG (optional)"""
    if not USE_3D_COORDS:
        return None
    
    try:
        mol_with_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 50
        
        status = AllChem.EmbedMolecule(mol_with_h, params=params)
        if status != 0:
            status = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            if status != 0:
                return None
        
        try:
            AllChem.MMFFOptimizeMolecule(mol_with_h)
        except:
            pass
        
        mol = Chem.RemoveHs(mol_with_h)
        if mol.GetNumConformers() == 0:
            return None
        
        conf = mol.GetConformer()
        coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]
        
        return torch.tensor(coords, dtype=torch.float)
    except:
        return None


def smiles_to_pyg_graph(smiles: str, target: float):
    """Convert SMILES + target to PyTorch Geometric Data object"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate 3D coordinates (optional)
    pos = generate_3d_coordinates(mol) if USE_3D_COORDS else None
    
    # Extract atom features (9D)
    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    if len(atom_features_list) == 0:
        return None
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
    # Extract edge features and build edge_index
    edge_index_list = []
    edge_attr_list = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = get_bond_features(bond)
        
        # Add distance feature if 3D coords available
        if pos is not None:
            dist = torch.norm(pos[i] - pos[j]).item()
            full_feat = np.concatenate([bond_feat, [dist]])
        else:
            full_feat = bond_feat
        
        # Add both directions (undirected graph)
        edge_index_list.append([i, j])
        edge_index_list.append([j, i])
        edge_attr_list.append(full_feat)
        edge_attr_list.append(full_feat)
    
    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long).contiguous()
        edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float)
    
    # Create target tensor
    y = torch.tensor([target], dtype=torch.float)
    
    # Build PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=x.size(0))
    
    # Add 3D positions if available
    if pos is not None:
        data.pos = pos
    
    return data


# NORMALIZATION (Computed from sample, not full dataset)

def compute_normalization_stats(sample_size=NORMALIZATION_SAMPLE_SIZE):
    """
    Compute normalization statistics from a sample of valid data.
    
    Memory-safe: Only loads sample_size samples, not full dataset.
    """
    print(f"\n Computing normalization statistics (sample of {sample_size:,})...")
    
    # Load assignments to get valid samples
    assignments_path = os.path.join(SPLIT_DIR, 'scaffold_assignments.csv')
    assignments = pd.read_csv(assignments_path)
    
    # Filter valid samples
    valid = assignments[
        (assignments['target'].notna()) & 
        (assignments['target'] > 0) & 
        (assignments['target'] <= 10)
    ].copy()
    
    # Sample if needed
    if len(valid) > sample_size:
        valid = valid.sample(n=sample_size, random_state=42)
    
    # Compute target normalization
    targets = valid['target'].values
    if TARGET_LOG_TRANSFORM:
        target_log = np.log1p(targets)
        target_mean = target_log.mean()
        target_std = target_log.std() + 1e-6
    else:
        target_mean = targets.mean()
        target_std = targets.std() + 1e-6
    
    # Sample node features for normalization
    all_node_feats = []
    for _, row in valid.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            for atom in mol.GetAtoms():
                all_node_feats.append(get_atom_features(atom))
    
    if all_node_feats:
        all_node_feats = np.array(all_node_feats)
        node_mean = all_node_feats.mean(axis=0)
        node_std = all_node_feats.std(axis=0) + 1e-6
    else:
        node_mean = np.zeros(ATOM_FEAT_DIM)
        node_std = np.ones(ATOM_FEAT_DIM)
    
    stats = {
        'target_mean': float(target_mean),
        'target_std': float(target_std),
        'node_mean': node_mean.tolist(),
        'node_std': node_std.tolist(),
        'log_transform': TARGET_LOG_TRANSFORM
    }
    
    # Save normalization stats
    stats_path = os.path.join(PREPROCESSED_DIR, 'normalization_stats.json')
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Target: mean={target_mean:.4f}, std={target_std:.4f}")
    print(f"  Node features: {ATOM_FEAT_DIM}D normalized")
    
    return stats


# STREAMING PREPROCESSING

def preprocess_streaming():
    """
    Preprocess dataset with streaming/chunk-wise processing.
    
    Memory-safe approach:
    - Read split assignments from disk (not full dataset)
    - Process one chunk at a time
    - Save output as chunked files per split
    - Never load full preprocessed data into memory
    """
    
    print("PREPROCESSING (STREAMING, MEMORY-SAFE, SPLIT-AWARE)")
    
    print(f"Input assignments: {SPLIT_DIR}/scaffold_assignments.csv")
    print(f"Output directory: {PREPROCESSED_DIR}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    
    
    # Load normalization stats
    norm_stats = compute_normalization_stats()
    
    # Load assignments
    assignments_path = os.path.join(SPLIT_DIR, 'scaffold_assignments.csv')
    assignments = pd.read_csv(assignments_path)
    print(f"\n Loaded {len(assignments):,} split assignments")
    
    # Statistics tracking
    stats = {split: {'processed': 0, 'failed': 0} for split in ['train', 'val', 'test']}
    stats['total'] = len(assignments)
    
    # Output file counters
    chunk_counters = {'train': 0, 'val': 0, 'test': 0}
    
    # Buffer for each split (cleared when reaching CHUNK_SIZE)
    buffers = {'train': [], 'val': [], 'test': []}
    
    # Process in chunks
    print(f"\n Converting SMILES → PyG graphs (chunked)...")
    
    for chunk_start in tqdm(range(0, len(assignments), CHUNK_SIZE), desc="Processing chunks"):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(assignments))
        chunk = assignments.iloc[chunk_start:chunk_end]
        
        for _, row in chunk.iterrows():
            split = row['split']
            smiles = row['smiles']
            target = row['target']
            
            # Convert to PyG graph
            try:
                data = smiles_to_pyg_graph(smiles, target)
                if data is None:
                    stats[split]['failed'] += 1
                    continue
                
                # Apply normalization
                if TARGET_LOG_TRANSFORM:
                    data.y = (torch.log1p(data.y) - norm_stats['target_mean']) / norm_stats['target_std']
                else:
                    data.y = (data.y - norm_stats['target_mean']) / norm_stats['target_std']
                
                node_mean = torch.tensor(norm_stats['node_mean'])
                node_std = torch.tensor(norm_stats['node_std'])
                data.x = (data.x - node_mean) / node_std
                
                # Add to buffer
                buffers[split].append(data)
                stats[split]['processed'] += 1
                
            except Exception as e:
                stats[split]['failed'] += 1
                continue
            
            # Save buffer if full
            if len(buffers[split]) >= CHUNK_SIZE:
                output_path = os.path.join(
                    PREPROCESSED_DIR, 
                    f"{split}_chunk_{chunk_counters[split]}.pt"
                )
                torch.save(buffers[split], output_path)
                chunk_counters[split] += 1
                buffers[split] = []  # Clear buffer
        
        # Free memory and run GC periodically
        del chunk
        if (chunk_start // CHUNK_SIZE) % GC_INTERVAL == 0:
            gc.collect()
    
    # Save remaining buffers
    print(f"\n Saving remaining buffers...")
    for split in ['train', 'val', 'test']:
        if len(buffers[split]) > 0:
            output_path = os.path.join(
                PREPROCESSED_DIR,
                f"{split}_chunk_{chunk_counters[split]}.pt"
            )
            torch.save(buffers[split], output_path)
            chunk_counters[split] += 1
            buffers[split] = []
    
    # Save preprocessing statistics
    stats_path = os.path.join(PREPROCESSED_DIR, 'preprocess_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    
    print("PREPROCESSING COMPLETE - SUMMARY")
    
    print(f"Total assignments: {stats['total']:,}")
    for split in ['train', 'val', 'test']:
        processed = stats[split]['processed']
        failed = stats[split]['failed']
        total = processed + failed
        print(f"{split.capitalize():5s}: {processed:,} processed, {failed:,} failed ({processed/total*100:.1f}% success)")
    
    print(f"\nOutput files:")
    for split in ['train', 'val', 'test']:
        count = chunk_counters[split]
        if count > 0:
            print(f"  {split}: {count} chunk files ({split}_chunk_0.pt ... {split}_chunk_{count-1}.pt)")
    
    print(f"\nNormalization stats: {PREPROCESSED_DIR}/normalization_stats.json")
    
    
    return stats, chunk_counters



def generate_chunk_metadata(chunk_counters: dict):
    """
    Generate metadata file for UltraFastDataset.
    Call this after preprocess_streaming() completes.
    """
    print(f"\n Generating chunk metadata for UltraFastDataset...")
    
    metadata = {}
    
    for split in ['train', 'val', 'test']:
        if chunk_counters.get(split, 0) == 0:
            continue
        
        chunk_info = []
        cumulative_idx = 0
        
        for chunk_idx in range(chunk_counters[split]):
            chunk_file = f"{split}_chunk_{chunk_idx}.pt"
            chunk_path = os.path.join(PREPROCESSED_DIR, chunk_file)
            
            if not os.path.exists(chunk_path):
                continue
            
            # Get chunk size
            chunk = torch.load(chunk_path, weights_only=False)
            size = len(chunk)
            
            chunk_info.append({
                'file': chunk_file,
                'start_idx': cumulative_idx,
                'size': size
            })
            
            cumulative_idx += size
            del chunk  
        
        if chunk_info:
            metadata[split] = chunk_info
            print(f"  {split}: {len(chunk_info)} chunks, {cumulative_idx:,} graphs")
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f" Metadata saved to: {METADATA_FILE}")
    return metadata


if __name__ == "__main__":
    stats, chunk_counts = preprocess_streaming()
    
    
    generate_chunk_metadata(chunk_counts)
    
    print(f"\n Preprocessing complete!")
    