"""Diagnostic script to check aggregated parquet files for NaN/Inf values.

Usage:
  python diagnose_nan.py --path /mnt/cellpainting-features/MorphoBenchFeaturesAggregated/non-patch/cpg-crispr
  python diagnose_nan.py --path /mnt/cellpainting-features/MorphoBenchFeaturesAggregated/non-patch/cpg-tgt2
  python diagnose_nan.py --pkl-path /mnt/cellpainting-features/MorphoBenchFeatures/non-patch/cpg-crispr/resnet_untrained --max-pkls 5
"""

import os
import sys
import glob
import argparse
import pickle
import numpy as np


def check_parquets(base_path):
    """Check all parquet files in a directory for NaN/Inf."""
    import pandas as pd
    
    parquet_files = sorted(glob.glob(os.path.join(base_path, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {base_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Checking {len(parquet_files)} parquet files in {base_path}")
    print(f"{'='*80}")
    
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        name = os.path.basename(pf)
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_df = df[numeric_cols]
        
        total_cells = numeric_df.size
        nan_count = numeric_df.isna().sum().sum()
        inf_count = np.isinf(numeric_df.values).sum() if total_cells > 0 else 0
        
        nan_pct = (nan_count / total_cells * 100) if total_cells > 0 else 0
        
        # Per-column NaN stats
        nan_per_col = numeric_df.isna().sum()
        cols_with_nan = nan_per_col[nan_per_col > 0]
        
        # Check data range
        mins = numeric_df.min().min()
        maxs = numeric_df.max().max()
        
        status = "✅ OK" if nan_count == 0 and inf_count == 0 else "❌ HAS NaN/Inf"
        
        print(f"\n{status} {name}")
        print(f"  Shape: {df.shape[0]} rows × {len(numeric_cols)} numeric cols")
        print(f"  NaN cells: {nan_count:,} / {total_cells:,} ({nan_pct:.2f}%)")
        print(f"  Inf cells: {inf_count:,}")
        print(f"  Value range: [{mins:.6f}, {maxs:.6f}]")
        
        if len(cols_with_nan) > 0:
            # Show which rows have NaN
            nan_rows = numeric_df.isna().any(axis=1).sum()
            print(f"  Rows with any NaN: {nan_rows} / {df.shape[0]}")
            print(f"  Columns with NaN: {len(cols_with_nan)} / {len(numeric_cols)}")
            
            # Check if it's ALL NaN for certain rows
            all_nan_rows = numeric_df.isna().all(axis=1).sum()
            print(f"  Rows that are ALL NaN: {all_nan_rows}")
            
            # Show first few NaN wells
            nan_mask = numeric_df.isna().any(axis=1)
            nan_wells = df.loc[nan_mask, ["Metadata_Plate", "Metadata_Well"]].head(10)
            print(f"  First NaN wells:")
            for _, row in nan_wells.iterrows():
                print(f"    plate={row['Metadata_Plate']} well={row['Metadata_Well']}")


def check_pkls(pkl_dir, max_pkls=5):
    """Check raw pkl files for NaN/Inf in features."""
    pattern = os.path.join(pkl_dir, "gpu_*_results", "results_part_*.pkl")
    pkl_files = sorted(glob.glob(pattern))
    
    if not pkl_files:
        print(f"No pkl files found matching {pattern}")
        return
    
    print(f"\n{'='*80}")
    print(f"Checking {min(max_pkls, len(pkl_files))} / {len(pkl_files)} pkl files in {pkl_dir}")
    print(f"{'='*80}")
    
    total_nan = 0
    total_inf = 0
    total_samples = 0
    
    for pf in pkl_files[:max_pkls]:
        with open(pf, 'rb') as f:
            data = pickle.load(f)
        
        batches = data if isinstance(data, list) else [data]
        
        for batch in batches:
            for key in ['features', 'cls_token_features', 'patch_token_features']:
                if key not in batch:
                    continue
                arr = batch[key]
                n = arr.shape[0]
                total_samples += n
                
                nan_mask = np.isnan(arr)
                inf_mask = np.isinf(arr)
                nan_count = nan_mask.any(axis=1).sum()
                inf_count = inf_mask.any(axis=1).sum()
                total_nan += nan_count
                total_inf += inf_count
                
                print(f"  {os.path.basename(pf)} [{key}]: shape={arr.shape}, dtype={arr.dtype}, "
                      f"nan_rows={nan_count}/{n}, inf_rows={inf_count}/{n}, "
                      f"range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
                
                if nan_count > 0:
                    # Show which samples have NaN
                    nan_indices = np.where(nan_mask.any(axis=1))[0][:3]
                    for idx in nan_indices:
                        fn = batch['filename'][idx] if isinstance(batch['filename'], list) else batch['filename']
                        pl = batch['plate_name'][idx] if isinstance(batch['plate_name'], list) else batch['plate_name']
                        print(f"    NaN sample: plate={pl}, file={fn}")
    
    print(f"\nSummary: {total_nan} NaN rows, {total_inf} Inf rows out of {total_samples} total samples")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NaN/Inf in aggregated or raw features")
    parser.add_argument("--path", type=str, help="Path to directory with aggregated parquet files")
    parser.add_argument("--pkl-path", type=str, help="Path to model dir with gpu_*_results/ pkl files")
    parser.add_argument("--max-pkls", type=int, default=5, help="Max pkl files to check")
    args = parser.parse_args()
    
    if args.path:
        check_parquets(args.path)
    
    if args.pkl_path:
        check_pkls(args.pkl_path, args.max_pkls)
    
    if not args.path and not args.pkl_path:
        print("Specify --path (for parquets) or --pkl-path (for raw pkls)")
        sys.exit(1)


if __name__ == "__main__":
    main()
