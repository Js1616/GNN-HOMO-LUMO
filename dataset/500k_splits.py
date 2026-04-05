import os
import gc
import random
import torch
import warnings
from pathlib import Path
from typing import List

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# Convert chunk to list
def _normalize_chunk_to_list(chunk):
    if isinstance(chunk, (list, tuple)):
        return list(chunk)
    if hasattr(chunk, "to_data_list"):
        return chunk.to_data_list()
    if hasattr(chunk, "num_graphs") and chunk.num_graphs > 1:
        try:
            from torch_geometric.utils import to_data_list
            return to_data_list(chunk)
        except ImportError:
            pass
    return [chunk]


# VALID GRAPH CHECK 
def is_valid_graph(data):
    try:
        # Node check
        if not hasattr(data, 'x') or data.x is None:
            return False
        if data.x.size(0) == 0:
            return False

        # Edge check
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            return False
        if data.edge_index.size(1) == 0:
            return False

        # Target check
        if not hasattr(data, 'y') or data.y is None:
            return False

        return True
    except:
        return False


# SAVE ONE SPLIT (500K)
def save_split(data, output_dir, split_id):
    print(f"\n Saving split_{split_id}...")

    total = len(data)

    train_size = int(0.875 * total)
    val_size = int(0.0625 * total)
    test_size = total - train_size - val_size

    random.shuffle(data)

    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]

    split_path = Path(output_dir) / f"split_{split_id}"
    split_path.mkdir(parents=True, exist_ok=True)

    torch.save(train, split_path / "train.pt")
    torch.save(val, split_path / "val.pt")
    torch.save(test, split_path / "test.pt")

    print(f" split_{split_id} saved:")
    print(f"   Train: {len(train):,}")
    print(f"   Val:   {len(val):,}")
    print(f"   Test:  {len(test):,}")


# MAIN FUNCTION
def create_multiple_500k_splits(
    chunk_dir: str,
    output_dir: str,
    split_size: int = 500_000,
    seed: int = 42
):

    random.seed(seed)

    chunk_path = Path(chunk_dir)
    chunk_files = sorted(chunk_path.glob("*.pt"))

    if not chunk_files:
        raise FileNotFoundError("No .pt files found")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    buffer = []
    total_loaded = 0
    split_id = 1
    skipped_files = 0
    skipped_graphs = 0

    print(f"[START] Found {len(chunk_files)} chunk files")

    for idx, chunk_file in enumerate(chunk_files, 1):

        try:
            chunk = torch.load(chunk_file, map_location="cpu", weights_only=False)
        except Exception:
            print(f" Skipping corrupted file: {chunk_file.name}")
            skipped_files += 1
            continue

        data_list = _normalize_chunk_to_list(chunk)

        # FILTER VALID GRAPHS
        valid_graphs = [g for g in data_list if is_valid_graph(g)]

        invalid_count = len(data_list) - len(valid_graphs)
        skipped_graphs += invalid_count

        if invalid_count > 0:
            print(f" Skipped {invalid_count} invalid graphs in {chunk_file.name}")

        buffer.extend(valid_graphs)
        total_loaded += len(valid_graphs)

        del chunk
        del data_list
        del valid_graphs

        if idx % 10 == 0:
            gc.collect()

        print(f" {idx}/{len(chunk_files)} -> Total valid graphs: {total_loaded:,}")

        # SAVE 500K SPLIT
        while len(buffer) >= split_size:
            current_split = buffer[:split_size]
            buffer = buffer[split_size:]

            save_split(current_split, output_dir, split_id)
            split_id += 1

            gc.collect()

    # SAVE REMAINING DATA
    if len(buffer) > 0:
        print("\n Saving remaining data as last split...")
        save_split(buffer, output_dir, split_id)

    print("\n DONE!")
    print(f"Total splits created: {split_id}")
    print(f"Skipped corrupted files: {skipped_files}")
    print(f"Skipped invalid graphs: {skipped_graphs}")
    print(f"Final usable graphs: {total_loaded:,}")


# RUN
if __name__ == "__main__":

    CHUNK_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\preprocessed"

    OUTPUT_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\500k_splits"

    create_multiple_500k_splits(
        chunk_dir=CHUNK_DIR,
        output_dir=OUTPUT_DIR,
        split_size=500_000
    )