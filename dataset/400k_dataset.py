import os
import gc
import random
import torch
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def _normalize_chunk_to_list(chunk: any) -> List:
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


def prepare_graph_subset(
    chunk_dir: str,
    output_dir: Optional[str] = None,
    target_samples: int = 400_000,
    train_size: int = 350_000,
    val_size: int = 25_000,
    test_size: int = 25_000,
    seed: int = 42
) -> Tuple[List, List, List]:

    if train_size + val_size + test_size != target_samples:
        raise ValueError("Split sizes must sum to target_samples")

    chunk_path = Path(chunk_dir)
    chunk_files = sorted(chunk_path.glob("*.pt"))

    output_path = Path(output_dir) if output_dir else chunk_path
    output_path.mkdir(parents=True, exist_ok=True)

    collected_data = []
    samples_loaded = 0
    skipped_files = 0

    print(f"[1/4] Found {len(chunk_files)} chunk files. Loading sequentially...")

    for file_idx, chunk_file in enumerate(chunk_files, 1):
        try:
            chunk = torch.load(chunk_file, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f" Skipping corrupted file: {chunk_file.name}")
            skipped_files += 1
            continue

        data_list = _normalize_chunk_to_list(chunk)
        num_graphs = len(data_list)

        collected_data.extend(data_list)
        samples_loaded += num_graphs

        del chunk
        del data_list

        if file_idx % 10 == 0:
            gc.collect()

        print(f"   {file_idx}/{len(chunk_files)}: {chunk_file.name} "
                f"({num_graphs} graphs -> Total: {samples_loaded:,})")

        if samples_loaded >= target_samples:
            collected_data = collected_data[:target_samples]
            samples_loaded = target_samples
            print(f"   Reached target of {target_samples:,}")
            break

    print(f"\n Skipped corrupted files: {skipped_files}")
    print(f" Total collected samples: {samples_loaded:,}")

    #  Handle if less data available
    if samples_loaded < target_samples:
        print(f" Adjusting splits because data < target")

        target_samples = samples_loaded
        train_size = int(0.875 * target_samples)
        val_size = int(0.0625 * target_samples)
        test_size = target_samples - train_size - val_size

    print(f"\n[2/4] Shuffling...")
    random.seed(seed)
    random.shuffle(collected_data)

    print("[3/4] Splitting dataset...")
    train_data = collected_data[:train_size]
    val_data = collected_data[train_size:train_size + val_size]
    test_data = collected_data[train_size + val_size:]

    del collected_data
    gc.collect()

    print("[4/4] Saving...")
    torch.save(train_data, output_path / "train.pt")
    torch.save(val_data, output_path / "val.pt")
    torch.save(test_data, output_path / "test.pt")

    print(f"\n DONE")
    print(f"Train: {len(train_data):,}")
    print(f"Val:   {len(val_data):,}")
    print(f"Test:  {len(test_data):,}")

    return train_data, val_data, test_data


if __name__ == "__main__":
    
    CHUNK_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\preprocessed"
    OUTPUT_DIR = r"C:\Users\js731\Downloads\GNN-HOMO-LUMO\dataset\400k_dataset"

    prepare_graph_subset(
        chunk_dir=CHUNK_DIR,
        output_dir=OUTPUT_DIR,  
        target_samples=400_000,
        train_size=350_000,
        val_size=25_000,
        test_size=25_000,
        seed=42
    )