# -*- coding: utf-8 -*-
"""
Splits existing training data (X_train.hkl, sources_train.hkl) into new
training and validation sets (90/10 split).

CRITICAL: The split is performed based on the 'sources' to ensure that all
frames from a particular source (e.g., a movie or clip) belong exclusively
to either the training or the validation set. This prevents data leakage.
"""
import os
import numpy as np
import hickle as hkl
from sklearn.model_selection import train_test_split # Used for splitting sources

# --- Configuration ---
# Path to the directory containing the original training files
DATA_DIR = '/local_raid1/03_user/jungmin/02_data/02_Movie/Concat6Movie/hkl/'
SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_concat6movie/hkl'

# Original training files
original_train_file = os.path.join(DATA_DIR, 'X_train.hkl')
original_sources_file = os.path.join(DATA_DIR, 'sources_train.hkl')

# Output files for the new split
new_train_file = os.path.join(SAVE_DIR, 'X_train_new.hkl')
new_sources_train_file = os.path.join(SAVE_DIR, 'sources_train_new.hkl')
val_file = os.path.join(SAVE_DIR, 'X_val_new.hkl') # Use a distinct name
sources_val_file = os.path.join(SAVE_DIR, 'sources_val_new.hkl') # Use a distinct name

# Validation set size (e.g., 0.1 for 10%)
validation_split_ratio = 0.1
random_seed = 42 # For reproducible splits
# --- End Configuration ---

if __name__ == "__main__":
    print("Starting data splitting process...")

    # --- 1. Load Original Data ---
    print("Loading original training data from: {}".format(DATA_DIR))
    if not os.path.exists(original_train_file) or not os.path.exists(original_sources_file):
        print("Error: Original training files ('X_train.hkl', 'sources_train.hkl') not found.")
        exit()

    try:
        X_train_all = hkl.load(original_train_file)
        sources_all_list = hkl.load(original_sources_file) # Load as potentially a list
        print("Data loaded successfully.")
        print(" Original X_train shape: {}".format(X_train_all.shape))
        # --- FIX: Use len() for list, convert to numpy array for processing ---
        sources_all = np.array(sources_all_list).flatten() # Convert list to numpy array
        print(" Original sources length: {}".format(len(sources_all_list))) # Use len() for original list
        # --- FIX END ---
    except Exception as e:
        print("Error loading hickle files: {}".format(e))
        exit()

    # Basic check: Number of frames must match
    if X_train_all.shape[0] != len(sources_all): # Use len() here too
        print("Error: Frame count mismatch (X: {}, sources: {}).".format(X_train_all.shape[0], len(sources_all)))
        exit()

    num_total_frames = X_train_all.shape[0]
    print("Total frames: {}".format(num_total_frames))

    # --- 2. Split Based on Sources ---
    print("\nSplitting sources into training ({:.0%}) and validation ({:.0%}) sets...".format(1-validation_split_ratio, validation_split_ratio))
    # Use the numpy array version for unique and isin
    unique_sources = np.unique(sources_all)
    num_unique_sources = len(unique_sources)
    print("Found {} unique sources.".format(num_unique_sources))

    if num_unique_sources <= 1:
        print("Warning: Only one source found. Splitting frames randomly (not recommended).")
        train_indices, val_indices = train_test_split(np.arange(num_total_frames), test_size=validation_split_ratio, random_state=random_seed)
    else:
        train_sources_list, val_sources_list = train_test_split(unique_sources, test_size=validation_split_ratio, random_state=random_seed)
        print(" Training sources count: {} (Sources: {})".format(len(train_sources_list), train_sources_list))
        print(" Validation sources count: {} (Sources: {})".format(len(val_sources_list), val_sources_list))

        # Use the numpy array version for boolean masking
        train_mask = np.isin(sources_all, train_sources_list)
        val_mask = np.isin(sources_all, val_sources_list)
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

    num_train_frames = len(train_indices)
    num_val_frames = len(val_indices)
    print(" Number of resulting training frames: {}".format(num_train_frames))
    print(" Number of resulting validation frames: {}".format(num_val_frames))

    if num_train_frames > 233892 and num_val_frames > 58472:
        print("Verification PASSED: New train/val sets are larger than the Titanic sets.")
    else:
        print("Verification WARNING: New train/val sets might not be larger than the Titanic sets.")


    # --- 3. Create New Datasets ---
    print("\nCreating new datasets in memory...")
    X_train_new = X_train_all[train_indices]
    # Use the numpy array version for indexing
    sources_train_new = sources_all[train_indices]
    X_val = X_train_all[val_indices]
    sources_val = sources_all[val_indices]

    del X_train_all, sources_all # Free memory

    # --- 4. Save New Datasets ---
    print("\nSaving split datasets (without compression)...")
    try:
        print(" Saving new training set X...")
        hkl.dump(X_train_new, new_train_file, mode='w')
        print(" Saving new training set sources...")
        # Save as numpy array for consistency, or convert back to list if preferred
        hkl.dump(sources_train_new, new_sources_train_file, mode='w')
        print(" Saving validation set X...")
        hkl.dump(X_val, val_file, mode='w')
        print(" Saving validation set sources...")
        hkl.dump(sources_val, sources_val_file, mode='w')
        print("\n Successfully saved:")
        print("  - Train X: {}".format(new_train_file))
        print("  - Train Sources: {}".format(new_sources_train_file))
        print("  - Val X: {}".format(val_file))
        print("  - Val Sources: {}".format(sources_val_file))
    except Exception as e:
        print("Error saving hickle files: {}".format(e))

    print("\nSplitting process complete.")