import numpy as np
import pandas as pd
import random

def run_mbb_resampling(df, target_col='PM10,24HOUR', block_size=24, threshold=155):
    # THE ULTIMATE LOCK: Lock both Python's built-in random AND numpy's random
    random.seed(42)
    np.random.seed(42)
    
    # Step 1: Create overlapping blocks
    def create_row_blocks(dataframe, b_size):
        blocks = []
        for i in range(len(dataframe) - b_size + 1):
            block = dataframe.iloc[i:i + b_size]
            blocks.append(block)
        return blocks

    blocks = create_row_blocks(df, block_size)

    # Step 2: Split blocks into extreme or normal
    extreme_blocks = []
    normal_blocks = []

    for block in blocks:
        if (block[target_col] >= threshold).any():
            extreme_blocks.append(block)
        else:
            normal_blocks.append(block)

    if len(extreme_blocks) == 0:
        return None, f"No extreme events found with threshold {threshold}."

    # Step 3: Resample 1:1 ratio
    n_extreme_sample = len(extreme_blocks)
    # Because random.seed(42) is set, these choices will be identical every time!
    resampled_extremes = random.choices(extreme_blocks, k=n_extreme_sample)
    resampled_normals = random.choices(normal_blocks, k=n_extreme_sample)

    # Step 4: Combine and shuffle
    resampled_blocks = resampled_extremes + resampled_normals
    # The shuffle will also happen in the exact same order every time!
    random.shuffle(resampled_blocks)

    balanced_df = pd.concat(resampled_blocks, ignore_index=True)
    return balanced_df, f"Success: Balanced dataset created ({len(balanced_df)} rows)."
