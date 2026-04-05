"""
=============================================================================
Production Configuration - Ultra-Fast Training Pipeline
=============================================================================

Optimizations:
- Auto-tuned multiprocessing settings per platform
- Memory-mapped file support for fast I/O
- Aggressive caching with smart eviction
- Minimal overhead monitoring
- Cross-platform compatibility

Performance targets:
- Batch loading: < 10ms overhead
- GPU utilization: > 95%
- Epoch time: 30min-2h for 3M graphs
- RAM usage: < 10GB stable

=============================================================================
"""

import os
import multiprocessing
import torch
from packaging import version

# PATHS

BASE_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Input/Output paths
INPUT_FILE = os.path.join(DATASET_DIR, "final_dataset_with_targets.pt")
SPLIT_DIR = os.path.join(DATASET_DIR, "scaffold_split")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "preprocessed")
METADATA_FILE = os.path.join(PREPROCESSED_DIR, "chunk_metadata.json")

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# PERFORMANCE SETTINGS - AUTO-TUNED

# System detection
IS_WINDOWS = os.name == 'nt'
IS_LINUX = os.name == 'posix'
HAS_GPU = torch.cuda.is_available()

# CPU/GPU configuration
NUM_CPU_CORES = multiprocessing.cpu_count()
GPU_COUNT = torch.cuda.device_count() if HAS_GPU else 0

# DataLoader optimization (auto-tuned)
if IS_WINDOWS:
    # Windows: conservative settings for stability
    NUM_WORKERS = 2
    PIN_MEMORY = True
    PERSISTENT_WORKERS = False  # Windows: persistent_workers can cause issues
    PREFETCH_FACTOR = 2
elif IS_LINUX:
    # Linux: aggressive settings for performance
    NUM_WORKERS = min(NUM_CPU_CORES // 2, 8)
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    PREFETCH_FACTOR = 3
else:
    # Fallback for other platforms
    NUM_WORKERS = 0
    PIN_MEMORY = False
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = None

MAX_TRAIN_SAMPLES = 300000 
# Memory management
MAX_RAM_GB = 10  # Target max RAM usage
CACHE_SIZE = 50  # LRU cache size for chunks (5-15 recommended)
USE_MMAP = True  # Use memory-mapped file loading for chunks

# Training optimization
BATCH_SIZE = 128  # Start with 32, adjust based on GPU memory
USE_AMP = HAS_GPU  # Mixed precision only on GPU
GRAD_CLIP_NORM = 1.0  # Gradient clipping threshold
GRAD_ACCUM_STEPS = 1  # Gradient accumulation steps (1 = no accumulation)

# Model settings
ATOM_FEAT_DIM = 9
BOND_FEAT_DIM = 3
USE_3D_COORDS = False
USE_POSITIONAL_ENCODING = False

CHUNK_SIZE = 5000   
GC_INTERVAL = 2 

# Normalization
TARGET_LOG_TRANSFORM = True
NORMALIZATION_SAMPLE_SIZE = 100000

# Monitoring (lightweight)
LOG_INTERVAL = 100  # Log performance every N batches
GPU_MEM_CHECK_INTERVAL = 500  # Check GPU memory every N batches

# Advanced optimizations

USE_TORCH_COMPILE = False
FUSED_OPTIMIZER = HAS_GPU  # Use fused AdamW if available

# RDKit SETTINGS

from rdkit import RDLogger
import warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')