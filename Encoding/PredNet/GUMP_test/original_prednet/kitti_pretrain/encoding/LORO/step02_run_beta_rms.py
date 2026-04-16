"""
Compute Beta RMS from W matrices for all subjects and layers.

Output: beta_rms_all_layers.npy (n_layers, n_voxels) per subject
        + group mean saved to .../group/beta_rms/
"""

import os
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_TYPE = "E"  # "Ahat" or "E"

BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4"

CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

N_LAYERS = 4

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_W_matrix(filepath):
    """
    Load W matrix from .h5 or .mat file.
    Returns W: (n_PCs, n_voxels)
    """
    filepath = Path(filepath)
    if filepath.suffix == '.h5':
        with h5py.File(filepath, 'r') as f:
            W = f['W'][:].astype(np.float32)
    elif filepath.suffix == '.mat':
        try:
            # v7.3 mat file (HDF5 format)
            with h5py.File(filepath, 'r') as f:
                W = f['W'][:].astype(np.float32)
        except:
            # older mat format
            data = sio.loadmat(filepath)
            W = data['W'].astype(np.float32)
    return W


def compute_beta_rms(W):
    """
    Compute RMS across PC dimension for each voxel.

    Args:
        W: (n_PCs, n_voxels)
    Returns:
        beta_rms: (n_voxels,)

    Why RMS instead of L2 norm:
    
        L1 has 248 PCs, L2-L4 have 500 PCs each.
        L2 norm = sqrt(sum(w^2)) grows with n_PCs -> unfair comparison across layers.
        RMS     = sqrt(mean(w^2)) divides by n_PCs -> fair comparison across layers.
    """
    return np.sqrt(np.mean(W ** 2, axis=0))


def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    """
    Save array as CIFTI dscalar file (cortex only).

    Args:
        data:          (n_layers, n_voxels) or (n_voxels,)
        template_file: path to CIFTI template
        output_file:   output path
        map_names:     list of map name strings
    Returns:
        True if saved successfully, False otherwise
    """
    template = nib.load(template_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_maps, n_voxels = data.shape

    brain_axis = template.header.get_axis(1)

    # collect cortex vertex indices only
    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))
    cortex_brain_axis = brain_axis[np.array(cortex_indices)]

    if n_voxels != len(cortex_indices):
        print("    WARNING: data voxels (%d) != cortex (%d)" % (n_voxels, len(cortex_indices)))
        return False

    if map_names is None:
        map_names = ['Layer%d' % (i + 1) for i in range(n_maps)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print(" Compute Beta RMS -- %s" % FEATURE_TYPE)
    print(" Base: %s" % BASE_DIR)
    print("=" * 70)

    SAVE_ROOT = Path(BASE_DIR) / FEATURE_TYPE
    GROUP_DIR = SAVE_ROOT / "group" / "beta_rms"
    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    all_subjects_rms = []  # accumulate per-subject arrays for group mean

    for s_idx, subj in enumerate(SUBJECTS):
        print("\n[%d/%d] %s" % (s_idx + 1, len(SUBJECTS), subj))

        subj_dir = SAVE_ROOT / subj
        if not subj_dir.exists():
            print("  [SKIP] Directory not found: %s" % subj_dir)
            continue

        beta_rms_layers = []

        for lay in range(1, N_LAYERS + 1):
            # try .h5 first, fall back to .mat
            w_path = subj_dir / ("W_layer%d.h5" % lay)
            if not w_path.exists():
                w_path = subj_dir / ("W_layer%d.mat" % lay)
            if not w_path.exists():
                print("  [MISSING] W_layer%d not found -> skipping subject" % lay)
                beta_rms_layers = []
                break

            W = load_W_matrix(w_path)
            rms = compute_beta_rms(W)  # (n_voxels,)
            beta_rms_layers.append(rms)

            print("  Layer %d: W=%s, n_PCs=%d, RMS mean=%.5f, std=%.5f"
                  % (lay, str(W.shape), W.shape[0], rms.mean(), rms.std()))

        if len(beta_rms_layers) != N_LAYERS:
            continue

        beta_rms_all = np.stack(beta_rms_layers)  # (n_layers, n_voxels)
        print("  beta_rms shape: %s" % str(beta_rms_all.shape))

        # --- save per-subject result ---
        out_npy = subj_dir / "beta_rms_all_layers.npy"
        np.save(str(out_npy), beta_rms_all)
        print("  Saved: %s" % out_npy.name)

        all_subjects_rms.append(beta_rms_all)

    # -------------------------------------------------------------------------
    # Group-level aggregation
    # -------------------------------------------------------------------------
    if len(all_subjects_rms) == 0:
        print("\n[ERROR] No subjects successfully processed.")
    else:
        n_valid = len(all_subjects_rms)
        group_stack = np.stack(all_subjects_rms)           # (n_subjects, n_layers, n_voxels)
        group_mean  = np.mean(group_stack, axis=0)         # (n_layers, n_voxels)
        group_sem   = np.std(group_stack, axis=0, ddof=1) / np.sqrt(n_valid)

        # save numpy arrays
        np.save(str(GROUP_DIR / "all_subjects_beta_rms.npy"), group_stack)
        np.save(str(GROUP_DIR / "group_mean_beta_rms.npy"),   group_mean)
        np.save(str(GROUP_DIR / "group_sem_beta_rms.npy"),    group_sem)
        print("\n[Group] Saved npy -> %s" % GROUP_DIR)

        # save CIFTI for wb_view visualization
        map_names = ["%s_L%d_RMS" % (FEATURE_TYPE, i + 1) for i in range(N_LAYERS)]
        cifti_out = str(GROUP_DIR / "group_mean_beta_rms.dscalar.nii")
        ok = save_cifti_dscalar(group_mean, CIFTI_TEMPLATE, cifti_out, map_names)
        if ok:
            print("[Group] Saved CIFTI -> group_mean_beta_rms.dscalar.nii")

        print("\n[Group] n_subjects=%d, shape=%s" % (n_valid, str(group_mean.shape)))
        print("[Group] Layer RMS summary:")
        for lay in range(N_LAYERS):
            print("  L%d: mean=%.5f, std=%.5f"
                  % (lay + 1, group_mean[lay].mean(), group_mean[lay].std()))

    print("\n" + "=" * 70)
    print(" Done!")
    print("=" * 70)