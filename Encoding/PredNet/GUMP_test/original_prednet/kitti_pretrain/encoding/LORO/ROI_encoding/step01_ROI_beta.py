"""
Step 1: ROI-Selective Beta Computation (PC1 on ROI-masked W)

Pipeline:
  1. Load ROI mask (data-driven parcels from create_ROI_mask_corr)
  2. For each subject & layer: subset W to ROI voxels only
  3. Compute PC1 on the ROI-masked W matrix
  4. Save subject-level ROI beta maps (+ whole-brain CIFTI with non-ROI = 0)

Rationale:
  Whole-brain PCA may be dominated by noise from non-responsive voxels.
  ROI-masked PCA captures genuine weight variation in encoding-significant regions.
"""

import os
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib
from sklearn.decomposition import PCA

# =============================================================================
# CONFIGURATION
# =============================================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4'
else:
    BASE_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4'

ROI_MASK_PATH = os.path.join(BASE_SAVE_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'ROI_mask_datadriven.npy')

CIFTI_TEMPLATE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'

SIGNAL_TYPES = ['Ahat', 'E']
N_LAYERS = 4

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

OUTPUT_DIR = os.path.join(BASE_SAVE_DIR, 'Results', 'Q1', 'ROI_encoding')

# =============================================================================
# HELPERS
# =============================================================================

def load_W_matrix(filepath):
    filepath = Path(filepath)
    if filepath.suffix == '.h5':
        with h5py.File(filepath, 'r') as f:
            W = f['W'][:].astype(np.float32)
    elif filepath.suffix == '.mat':
        try:
            with h5py.File(filepath, 'r') as f:
                W = f['W'][:].astype(np.float32)
        except Exception:
            data = sio.loadmat(filepath)
            W = data['W'].astype(np.float32)
    else:
        raise ValueError(f'Unknown format: {filepath.suffix}')
    return W


def compute_beta_pc1(W):
    """PCA on W.T -> PC1 scores as beta."""
    W_T = W.T  # (n_voxels, n_features)
    W_centered = W_T - W_T.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(W_centered).flatten()
    return pc1_scores, pca.explained_variance_ratio_[0]


def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    template = nib.load(template_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_maps, n_data = data.shape
    brain_axis = template.header.get_axis(1)
    ci = []
    for name, slc, _ in brain_axis.iter_structures():
        if 'CORTEX' in name:
            ci.extend(range(slc.start, slc.stop))
    cortex_axis = brain_axis[np.array(ci)]
    if n_data != len(ci):
        print(f'    WARNING: data ({n_data}) != cortex ({len(ci)})')
        return False
    if map_names is None:
        map_names = [f'Map_{i+1}' for i in range(n_maps)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_axis))
    img = nib.Cifti2Image(data.astype(np.float32), header=hdr)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(img, output_file)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print(' ROI-Selective Beta Computation (PC1 on ROI-masked W)')
    print('=' * 70)

    # Load ROI mask
    print('\n[1] Loading ROI mask...')
    roi_mask = np.load(ROI_MASK_PATH).astype(np.int32)
    roi_indices = np.where(roi_mask > 0)[0]
    n_cortex = len(roi_mask)
    n_roi = len(roi_indices)
    print(f'    ROI voxels: {n_roi} / {n_cortex} ({100*n_roi/n_cortex:.1f}%)')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'roi_indices.npy'), roi_indices)

    for sig_type in SIGNAL_TYPES:
        print(f'\n{"="*70}')
        print(f' Processing: {sig_type}')
        print(f'{"="*70}')

        sig_dir = os.path.join(OUTPUT_DIR, sig_type)
        os.makedirs(sig_dir, exist_ok=True)

        all_subject_betas = []  # (n_subjects, n_layers, n_roi)

        for si, subject in enumerate(SUBJECTS):
            print(f'\n  [{si+1}/{len(SUBJECTS)}] {subject}')
            subj_w_dir = Path(BASE_SAVE_DIR) / sig_type / subject

            betas_layers = []
            exp_vars = []

            for lay in range(N_LAYERS):
                w_path = subj_w_dir / f'W_layer{lay+1}.h5'
                if not w_path.exists():
                    print(f'    [ERROR] {w_path} not found')
                    break

                W = load_W_matrix(w_path)  # (n_features, n_cortex)

                # Subset to ROI voxels
                W_roi = W[:, roi_indices]  # (n_features, n_roi)

                # PC1 on ROI
                beta_roi, exp_var = compute_beta_pc1(W_roi)
                betas_layers.append(beta_roi)
                exp_vars.append(exp_var)

                print(f'    Layer {lay+1}: W={W.shape} -> W_roi={W_roi.shape} '
                      f'-> beta_roi={beta_roi.shape}, PC1 var={exp_var:.2%}')

            if len(betas_layers) != N_LAYERS:
                print(f'    [SKIP] incomplete layers')
                continue

            beta_stack = np.stack(betas_layers)  # (n_layers, n_roi)
            all_subject_betas.append(beta_stack)

            # Save per-subject ROI beta
            subj_out = os.path.join(sig_dir, subject)
            os.makedirs(subj_out, exist_ok=True)
            np.save(os.path.join(subj_out, 'beta_pc1_ROI.npy'), beta_stack)

            # Save whole-brain CIFTI (non-ROI = 0)
            beta_brain = np.zeros((N_LAYERS, n_cortex), dtype=np.float32)
            beta_brain[:, roi_indices] = beta_stack
            map_names = [f'{sig_type}{i}_beta_ROI' for i in range(N_LAYERS)]
            save_cifti_dscalar(beta_brain, CIFTI_TEMPLATE,
                               os.path.join(subj_out, 'beta_pc1_ROI.dscalar.nii'),
                               map_names)

            print(f'    Mean PC1 var: {np.mean(exp_vars):.2%}')
            for i in range(N_LAYERS):
                print(f'    Layer {i+1}: mean={beta_stack[i].mean():.4f}, '
                      f'std={beta_stack[i].std():.4f}')

        # Save stacked array for group analysis
        if all_subject_betas:
            stacked = np.stack(all_subject_betas)  # (n_subjects, n_layers, n_roi)
            np.save(os.path.join(sig_dir, 'all_subjects_beta_ROI.npy'), stacked)
            print(f'\n  Saved all_subjects_beta_ROI.npy: {stacked.shape}')

    print(f'\n{"="*70}')
    print(f' Done! Output: {OUTPUT_DIR}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
