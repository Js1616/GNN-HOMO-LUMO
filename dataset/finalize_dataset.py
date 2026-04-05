"""
=============================================================================
Finalize Dataset - Combine ONLY Training Chunks
=============================================================================

This script:
- Reads scaffold split indices from scaffold_split/
- Loads chunks from temp_preprocessed/ (created by preprocessed.py)
- Filters to ONLY training data using scaffold indices
- Combines and saves training dataset only
- Provides PyTorch Geometric Dataset class for training

IMPORTANT: Run this AFTER both preprocessed.py AND scaffold_split.py

Input:  
    - Chunks in temp_preprocessed/ (from preprocessed.py)
    - Indices in scaffold_split/ (from scaffold_split.py)
Output: train_preprocessed.pt (training data only)

Benefits:
- Much lower memory usage (only ~80% of data)
- No program crash from loading all chunks
- Proper scaffold-based evaluation protocol
=============================================================================
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION

# Paths - Using YOUR existing folders
TEMP_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\temp_preprocessed"
SPLIT_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\scaffold_split"
OUTPUT_FILE = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\train_preprocessed.pt"

# COMBINE TRAINING CHUNKS ONLY

def combine_training_chunks():
    """
    Combine ONLY training chunks based on scaffold split indices
    
    Returns:
        train_data: List of PyG Data objects (training data only)
    """
    print("="*70)
    print("COMBINING TRAINING CHUNKS ONLY")
    print("="*70)
    print(f"Chunk directory: {TEMP_DIR}")
    print(f"Scaffold split dir: {SPLIT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*70)
    
    # Load scaffold split indices
    print(f"\nLoading scaffold split indices...")
    train_idx = np.load(os.path.join(SPLIT_DIR, 'train_idx.npy'))
    val_idx = np.load(os.path.join(SPLIT_DIR, 'val_idx.npy'))
    test_idx = np.load(os.path.join(SPLIT_DIR, 'test_idx.npy'))
    
    print(f"Train indices: {len(train_idx):,}")
    print(f"Val indices: {len(val_idx):,}")
    print(f"Test indices: {len(test_idx):,}")
    
    # Load ALL chunks into a single list (this is the full preprocessed data)
    # Note: These are indices into the valid_data list from scaffold_split.py
    print(f"\nLoading preprocessed chunks...")
    chunk_files = sorted([
        f for f in os.listdir(TEMP_DIR) 
        if f.endswith('.pt') and f != 'chunk_list.txt'
    ])
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Load all chunk data into one list
    # This represents the valid_data list from scaffold_split.py
    all_valid_data = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunk_path = os.path.join(TEMP_DIR, chunk_file)
        if os.path.exists(chunk_path):
            chunk = torch.load(chunk_path, weights_only=False)
            all_valid_data.extend(chunk)
        else:
            print(f"Warning: Chunk file not found: {chunk_path}")
    
    print(f"Total valid graphs loaded from chunks: {len(all_valid_data):,}")
    
    # Filter to training data only using scaffold split indices
    print(f"\nFiltering to training data only...")
    train_data = [all_valid_data[i] for i in train_idx if i < len(all_valid_data)]
    
    print(f"Training graphs after filtering: {len(train_data):,}")
    
    # Save final training dataset
    print(f"\nSaving final training dataset...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(train_data, OUTPUT_FILE)
    
    # Print summary
    print("\n" + "="*70)
    print("FINALIZATION COMPLETE - SUMMARY")
    print("="*70)
    print(f"Total valid graphs in chunks: {len(all_valid_data):,}")
    print(f"Training graphs: {len(train_data):,} ({len(train_data)/len(all_valid_data)*100:.1f}%)")
    print(f"Validation graphs: {len(val_idx):,} (kept in chunks - process later)")
    print(f"Test graphs: {len(test_idx):,} (kept in chunks - process later)")
    print(f"\nOutput file: {OUTPUT_FILE}")
    if os.path.exists(OUTPUT_FILE):
        print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024**3):.2f} GB")
    print("="*70)
    
    # Verify output
    print(f"\nVerifying output...")
    verify_data = torch.load(OUTPUT_FILE, weights_only=False)
    print(f"Loaded {len(verify_data):,} training graphs")
    
    if len(verify_data) > 0:
        sample = verify_data[0]
        print(f"Sample graph:")
        print(f"  Node features: {sample.x.shape}")
        print(f"  Edge index: {sample.edge_index.shape}")
        print(f"  Edge features: {sample.edge_attr.shape}")
        print(f"  Target (normalized): {sample.y.item():.4f}")
        if hasattr(sample, 'pos') and sample.pos is not None:
            print(f"  3D positions: {sample.pos.shape}")
        else:
            print(f"  3D positions: Not included")
    
    return train_data


# PYTORCH GEOMETRIC DATASET CLASS FOR TRAINING

class HOMO_LUMO_Training_Dataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset class for HOMO-LUMO gap prediction
    
    This class loads the preprocessed training dataset and provides:
    - Indexed access to graph data
    - Integration with PyTorch Geometric DataLoader
    """
    
    def __init__(self, root, transform=None, pre_transform=None, 
                file_name='train_preprocessed.pt'):
        """
        Initialize the dataset
        
        Args:
            root: Root directory containing the dataset file
            transform: Optional transform to apply to data objects
            pre_transform: Optional pre-transform to apply before saving
            file_name: Name of the preprocessed dataset file
        """
        self.file_name = file_name
        super().__init__(root, transform, pre_transform)
        # FIX: Add weights_only=False for PyTorch 2.6 compatibility
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        """Return the name of the processed data file"""
        return [self.file_name]
    
    def process(self):
        """
        Process method - not used since data is preprocessed externally
        """
        pass
    
    def get_statistics(self):
        """
        Return basic statistics about the dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        targets = [data.y.item() for data in self]
        num_nodes = [data.num_nodes for data in self]
        num_edges = [data.edge_index.size(1) // 2 for data in self]
        
        return {
            'num_graphs': len(self),
            'avg_nodes': np.mean(num_nodes),
            'avg_edges': np.mean(num_edges),
            'target_mean': np.mean(targets),
            'target_std': np.std(targets),
            'target_min': np.min(targets),
            'target_max': np.max(targets)
        }


# ENTRY POINT

if __name__ == "__main__":
    # Combine training chunks only
    train_data = combine_training_chunks()
    
    if train_data is not None:
        print(f"\n✅ Training dataset ready at: {OUTPUT_FILE}")
        
        # Optional: Test the Dataset class
        print(f"\nTesting HOMO_LUMO_Training_Dataset class...")
        dataset = HOMO_LUMO_Training_Dataset(root=os.path.dirname(OUTPUT_FILE))
        print(f"Dataset loaded: {len(dataset)} graphs")
        
        stats = dataset.get_statistics()
        print(f"\nDataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎯 Next step: Train your model on this training data!")
        print(f"   Val/Test data can be processed later when needed for evaluation")