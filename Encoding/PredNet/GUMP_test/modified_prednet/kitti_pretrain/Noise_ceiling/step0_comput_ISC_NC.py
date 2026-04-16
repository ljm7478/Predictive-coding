"""
Step 0: Compute ISC (Noise Ceiling) for Encoding Analysis
==========================================================

ISC (Inter-Subject Correlation) = Noise Ceiling

Method: Leave-One-Out ISC

Output:
- isc_map.dscalar.nii: Voxelwise ISC map
- isc_map.npy: For further analysis
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import stats
import h5py
import scipy.io as sio
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'budapest' or 'gump'
DATASET = 'gump'

if DATASET == 'budapest':
    FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/fmri_new"
    OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/modified_prednet/kitti_pretrain_result/LORO_PCAcap/layer4/divergence_analysis/step0_noise_ceiling"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
    SUBJECTS = [
        "sub-sid000005", "sub-sid000007", "sub-sid000009", "sub-sid000010", "sub-sid000013",
        "sub-sid000020", "sub-sid000021", "sub-sid000024", "sub-sid000025", "sub-sid000029",
        "sub-sid000030", "sub-sid000034", "sub-sid000050", "sub-sid000052", "sub-sid000055",
        "sub-sid000114", "sub-sid000120", "sub-sid000134", "sub-sid000142", "sub-sid000278",
        "sub-sid000416", "sub-sid000499", "sub-sid000522", "sub-sid000535", "sub-sid000560"
    ]
    # Budapest: 5 runs, use test runs (run4, run5) for ISC
    # Note: stored as fmri_data_per_run.run1, .run2, ... in MATLAB struct
    TEST_RUNS = ['run4', 'run5']
    
else:  # gump
    FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new"
    OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/kitti_pretrain_result/LORO_PCAcap/layer4/divergence_analysis/step0_noise_ceiling"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"
    SUBJECTS = [
        "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
        "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
        "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
    ]
    # GUMP: 8 runs, use test runs (run5-run8) for ISC
    TEST_RUNS = ['run5', 'run6', 'run7', 'run8']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_fmri_per_run(fmri_path):
    """
    Load per-run fMRI data from .mat file (MATLAB -v7.3 = HDF5)
    
    Structure: fmri_data_per_run/run1, run2, ... 
    HDF5 stores as (n_timepoints, n_voxels) - already correct!
    
    Returns: dict with run names as keys, (n_timepoints, n_voxels) as values
    """
    if not Path(fmri_path).exists():
        print(f"    File not found: {fmri_path}")
        return None
    
    data = {}
    
    try:
        with h5py.File(fmri_path, 'r') as f:
            # Access fmri_data_per_run group
            if 'fmri_data_per_run' in f:
                grp = f['fmri_data_per_run']
                for run_name in grp.keys():
                    if run_name.startswith('run'):
                        # Shape is (TRs, voxels) - keep as is!
                        run_data = grp[run_name][:]
                        data[run_name] = run_data.astype(np.float32)  # (T, V)
            
            if len(data) > 0:
                return data
                
    except Exception as e:
        print(f"    HDF5 error: {e}")
    
    # Fallback to scipy.io (older MATLAB format)
    try:
        mat_data = sio.loadmat(fmri_path)
        
        if 'fmri_data_per_run' in mat_data:
            struct = mat_data['fmri_data_per_run']
            if hasattr(struct, 'dtype') and struct.dtype.names:
                for name in struct.dtype.names:
                    if name.startswith('run'):
                        data[name] = struct[name][0, 0].astype(np.float32)
        
        if len(data) > 0:
            return data
            
    except Exception as e:
        print(f"    scipy.io error: {e}")
    
    return None if len(data) == 0 else data


def compute_isc_leave_one_out(fmri_all_subjects):
    """
    Compute Leave-One-Out ISC
    
    Args:
        fmri_all_subjects: (n_subjects, n_timepoints, n_voxels)
    
    Returns:
        isc: (n_voxels,) ISC values
    """
    n_subjects, n_timepoints, n_voxels = fmri_all_subjects.shape
    
    print(f"    Computing Leave-One-Out ISC...")
    print(f"    Shape: {n_subjects} subjects ¡¿ {n_timepoints} TRs ¡¿ {n_voxels} voxels")
    
    isc_per_subject = np.zeros((n_subjects, n_voxels), dtype=np.float32)
    
    for subj in tqdm(range(n_subjects), desc="    ISC"):
        # Leave one out
        left_out = fmri_all_subjects[subj]  # (T, V)
        
        # Average of others
        others_idx = np.arange(n_subjects) != subj
        others_mean = np.mean(fmri_all_subjects[others_idx], axis=0)  # (T, V)
        
        # Correlation per voxel (vectorized)
        # Standardize
        left_out_z = (left_out - np.mean(left_out, axis=0, keepdims=True)) / (np.std(left_out, axis=0, keepdims=True) + 1e-10)
        others_z = (others_mean - np.mean(others_mean, axis=0, keepdims=True)) / (np.std(others_mean, axis=0, keepdims=True) + 1e-10)
        
        # Correlation = mean of element-wise product of z-scores
        corr = np.mean(left_out_z * others_z, axis=0)
        
        isc_per_subject[subj] = corr
    
    # Average across subjects
    isc = np.mean(isc_per_subject, axis=0)
    
    return isc, isc_per_subject


def save_cifti_dscalar(data, template_file, output_file, map_names):
    """Save as CIFTI dscalar"""
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_voxels = data.shape
    brain_axis = template.header.get_axis(1)
    n_brain = brain_axis.size
    
    if n_voxels != n_brain:
        if n_voxels > n_brain:
            data = data[:, :n_brain]
        else:
            padded = np.zeros((n_maps, n_brain), dtype=np.float32)
            padded[:, :n_voxels] = data
            data = padded
    
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    nib.save(new_img, str(output_file))
    print(f"    Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print(f" STEP 0: COMPUTE ISC (NOISE CEILING) - {DATASET.upper()}")
    print("="*70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load fMRI data for all subjects
    # =========================================================================
    print("\n[1] Loading fMRI data...")
    
    all_fmri_data = []
    valid_subjects = []
    
    for subject in SUBJECTS:
        fmri_path = Path(FMRI_DATA_ROOT) / subject / f"{subject}_per_run_fmri.mat"
        
        fmri_data = load_fmri_per_run(str(fmri_path))
        
        if fmri_data is None:
            print(f"    [SKIP] {subject}: Data not found")
            continue
        
        # Show available runs
        available_runs = list(fmri_data.keys())
        print(f"    {subject}: Available runs = {available_runs}")
        
        # Concatenate test runs
        run_data = []
        for run_name in TEST_RUNS:
            if run_name in fmri_data:
                run_data.append(fmri_data[run_name])
            else:
                print(f"    [WARN] {subject}: {run_name} not found")
        
        if len(run_data) == 0:
            print(f"    [SKIP] {subject}: No test runs found")
            continue
        
        # Data is (T, V) for each run, concatenate along time axis
        concatenated = np.concatenate(run_data, axis=0)  # (T_total, V)
        
        all_fmri_data.append(concatenated)
        valid_subjects.append(subject)
        
        print(f"    {subject}: {concatenated.shape[0]} TRs ¡¿ {concatenated.shape[1]} voxels")
    
    if len(all_fmri_data) == 0:
        print("\nERROR: No fMRI data loaded!")
        return
    
    # Check all subjects have same shape
    shapes = [d.shape for d in all_fmri_data]
    if len(set(shapes)) > 1:
        print(f"\n    [WARN] Different shapes detected: {set(shapes)}")
        # Use minimum timepoints
        min_t = min(s[0] for s in shapes)
        all_fmri_data = [d[:min_t, :] for d in all_fmri_data]
        print(f"    Truncated to {min_t} timepoints")
    
    # Stack: (n_subjects, n_timepoints, n_voxels)
    fmri_stack = np.stack(all_fmri_data, axis=0)
    n_subjects, n_timepoints, n_voxels = fmri_stack.shape
    
    print(f"\n    Final shape: {n_subjects} subjects ¡¿ {n_timepoints} TRs ¡¿ {n_voxels} voxels")
    print(f"    Valid subjects: {valid_subjects}")
    
    # =========================================================================
    # Compute ISC
    # =========================================================================
    print("\n[2] Computing ISC (Leave-One-Out)...")
    
    isc, isc_per_subject = compute_isc_leave_one_out(fmri_stack)
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n[3] ISC Summary Statistics:")
    print(f"    Mean ISC: {np.nanmean(isc):.4f}")
    print(f"    Median ISC: {np.nanmedian(isc):.4f}")
    print(f"    Std ISC: {np.nanstd(isc):.4f}")
    print(f"    Range: [{np.nanmin(isc):.4f}, {np.nanmax(isc):.4f}]")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print(f"\n    Percentiles:")
    for p in percentiles:
        val = np.nanpercentile(isc, p)
        print(f"      {p}%: {val:.4f}")
    
    # Reliable voxels (ISC > threshold)
    thresholds = [0.1, 0.2, 0.3]
    print(f"\n    Reliable voxels:")
    for thresh in thresholds:
        n_reliable = np.sum(isc > thresh)
        pct = 100 * n_reliable / n_voxels
        print(f"      ISC > {thresh}: {n_reliable:,} ({pct:.1f}%)")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n[4] Saving results...")
    
    # Save as numpy
    np.save(output_dir / "isc_map.npy", isc)
    np.save(output_dir / "isc_per_subject.npy", isc_per_subject)
    print(f"    isc_map.npy: {isc.shape}")
    print(f"    isc_per_subject.npy: {isc_per_subject.shape}")
    
    # Save as CIFTI
    save_cifti_dscalar(isc, CIFTI_TEMPLATE, 
                       str(output_dir / "isc_map.dscalar.nii"),
                       ["ISC"])
    
    # Save reliable mask (ISC > 0.1)
    reliable_mask = (isc > 0.1).astype(np.float32)
    save_cifti_dscalar(reliable_mask, CIFTI_TEMPLATE,
                       str(output_dir / "reliable_mask_isc01.dscalar.nii"),
                       ["ISC>0.1"])
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" STEP 0 COMPLETE")
    print("="*70)
    print(f"""
    Dataset: {DATASET.upper()}
    Subjects: {n_subjects}
    Timepoints: {n_timepoints}
    Voxels: {n_voxels}
    
    ISC Summary:
      Mean: {np.nanmean(isc):.4f}
      Median: {np.nanmedian(isc):.4f}
      Reliable voxels (ISC > 0.1): {np.sum(isc > 0.1):,} ({100*np.sum(isc > 0.1)/n_voxels:.1f}%)
    
    Output files:
      - isc_map.npy
      - isc_per_subject.npy
      - isc_map.dscalar.nii
      - reliable_mask_isc01.dscalar.nii
    
    Output directory: {output_dir}
    
    Next step: Step 1 - Normalize beta/corr by ISC
    """)


if __name__ == "__main__":
    main()