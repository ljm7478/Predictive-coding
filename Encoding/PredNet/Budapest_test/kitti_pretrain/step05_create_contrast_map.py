"""
Step 1: Create Prediction - Error Contrast Map per Layer (BUDAPEST)
====================================================================
Output: contrast_per_layer.dscalar.nii (4 maps: L1, L2, L3, L4)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import h5py
import scipy.io as sio

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DATA_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"

OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis/step1_contrast_maps"

CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"

SUBJECTS = [
    "sub-sid000005", "sub-sid000007", "sub-sid000009", "sub-sid000010", "sub-sid000013",
    "sub-sid000020", "sub-sid000021", "sub-sid000024", "sub-sid000025", "sub-sid000029",
    "sub-sid000030", "sub-sid000034", "sub-sid000050", "sub-sid000052", "sub-sid000055",
    "sub-sid000114", "sub-sid000120", "sub-sid000134", "sub-sid000142", "sub-sid000278",
    "sub-sid000416", "sub-sid000499", "sub-sid000522", "sub-sid000535", "sub-sid000560"
]

LAYER_NAMES = ['L1', 'L2', 'L3', 'L4']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_subject_beta(subject_dir):
    """Load beta_pc1_all_layers.npy"""
    npy_path = subject_dir / 'beta_pc1_all_layers.npy'
    mat_path = subject_dir / 'beta_pc1_all_layers.mat'
    
    if npy_path.exists():
        return np.load(str(npy_path))
    elif mat_path.exists():
        try:
            with h5py.File(mat_path, 'r') as f:
                for key in f.keys():
                    if not key.startswith('#'):
                        return f[key][:].astype(np.float32)
        except:
            data = sio.loadmat(str(mat_path))
            for key in data.keys():
                if not key.startswith('_'):
                    return data[key].astype(np.float32)
    return None


def load_all_subjects(base_dir, subjects, feature_type):
    """Load all subjects' beta data"""
    save_root = Path(base_dir) / feature_type
    all_data = []
    valid_subjects = []
    
    for subject in subjects:
        subject_dir = save_root / subject
        data = load_subject_beta(subject_dir)
        if data is not None:
            all_data.append(data)
            valid_subjects.append(subject)
    
    if len(all_data) == 0:
        return None, []
    
    return np.stack(all_data, axis=0), valid_subjects  # (n_subjects, n_layers, n_voxels)


def save_cifti_dscalar(data, template_file, output_file, map_names):
    """Save as CIFTI dscalar"""
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_voxels = data.shape
    
    # Get brain axis from template
    brain_axis = template.header.get_axis(1)
    
    # Adjust data size if needed
    n_brain = brain_axis.size
    if n_voxels != n_brain:
        if n_voxels > n_brain:
            data = data[:, :n_brain]
        else:
            padded = np.zeros((n_maps, n_brain), dtype=np.float32)
            padded[:, :n_voxels] = data
            data = padded
    
    # Create scalar axis
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    
    # Create new header and image
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    nib.save(new_img, str(output_file))
    print(f"  Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print(" STEP 1: CREATE PREDICTION - ERROR CONTRAST MAPS (BUDAPEST)")
    print("="*70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[1] Loading data...")
    
    ahat_beta, valid_subjects = load_all_subjects(BASE_DATA_DIR, SUBJECTS, "Ahat")
    e_beta, _ = load_all_subjects(BASE_DATA_DIR, SUBJECTS, "E")
    
    if ahat_beta is None:
        print("ERROR: Data not found!")
        return
    
    n_subjects, n_layers, n_voxels = ahat_beta.shape
    print(f"    Subjects: {n_subjects} / {len(SUBJECTS)}")
    print(f"    Layers: {n_layers}")
    print(f"    Voxels: {n_voxels}")
    print(f"    Valid subjects: {valid_subjects[:5]}...")
    
    # =========================================================================
    # Compute Group Mean
    # =========================================================================
    print("\n[2] Computing group mean...")
    
    ahat_group = np.nanmean(ahat_beta, axis=0)  # (4, n_voxels)
    e_group = np.nanmean(e_beta, axis=0)        # (4, n_voxels)
    
    print(f"    Ahat range: [{ahat_group.min():.4f}, {ahat_group.max():.4f}]")
    print(f"    E range: [{e_group.min():.4f}, {e_group.max():.4f}]")
    
    # =========================================================================
    # Compute Contrast: Ahat - E per Layer
    # =========================================================================
    print("\n[3] Computing contrast (Ahat - E) per layer...")
    
    contrast = ahat_group - e_group  # (4, n_voxels)
    
    print(f"\n    {'Layer':<8} {'Min':>12} {'Max':>12} {'Mean':>12}")
    print(f"    {'-'*48}")
    for i, layer in enumerate(LAYER_NAMES):
        print(f"    {layer:<8} {contrast[i].min():>+12.4f} {contrast[i].max():>+12.4f} {contrast[i].mean():>+12.4f}")
    
    # =========================================================================
    # Save CIFTI Files
    # =========================================================================
    print("\n[4] Saving CIFTI files...")
    
    # 1. Contrast per layer (main output)
    save_cifti_dscalar(
        contrast, CIFTI_TEMPLATE,
        str(output_dir / "contrast_AhatMinusE_per_layer.dscalar.nii"),
        ["Contrast_L1", "Contrast_L2", "Contrast_L3", "Contrast_L4"]
    )
    
    # 2. Ahat per layer
    save_cifti_dscalar(
        ahat_group, CIFTI_TEMPLATE,
        str(output_dir / "ahat_beta_per_layer.dscalar.nii"),
        ["Ahat_L1", "Ahat_L2", "Ahat_L3", "Ahat_L4"]
    )
    
    # 3. E per layer
    save_cifti_dscalar(
        e_group, CIFTI_TEMPLATE,
        str(output_dir / "e_beta_per_layer.dscalar.nii"),
        ["E_L1", "E_L2", "E_L3", "E_L4"]
    )
    
    # 4. L4 only (strongest effect expected)
    save_cifti_dscalar(
        contrast[3:4], CIFTI_TEMPLATE,
        str(output_dir / "contrast_L4_only.dscalar.nii"),
        ["Contrast_L4"]
    )
    
    # 5. L4 - L1 difference (hierarchy effect)
    hierarchy_effect = contrast[3] - contrast[0]  # How contrast changes from L1 to L4
    save_cifti_dscalar(
        hierarchy_effect.reshape(1, -1), CIFTI_TEMPLATE,
        str(output_dir / "hierarchy_effect_L4minusL1.dscalar.nii"),
        ["Hierarchy_L4minusL1"]
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print(" OUTPUT FILES")
    print("="*70)
    print(f"""
    Directory: {output_dir}
    
    1. contrast_AhatMinusE_per_layer.dscalar.nii  ¡ç MAIN: L1~L4 contrast
       - RED: Ahat > E (Prediction Enhanced)
       - BLUE: Ahat < E (Prediction Suppressed)
    
    2. contrast_L4_only.dscalar.nii               ¡ç L4 only (strongest)
    
    3. hierarchy_effect_L4minusL1.dscalar.nii     ¡ç L4-L1 (hierarchy effect)
       - BLUE: Contrast decreases (more suppression in L4)
       - RED: Contrast increases (more enhancement in L4)
    
    4. ahat_beta_per_layer.dscalar.nii            ¡ç Ahat only
    5. e_beta_per_layer.dscalar.nii               ¡ç E only
    
    WB_VIEW INSTRUCTIONS:
    - Open contrast_AhatMinusE_per_layer.dscalar.nii
    - Use ROY-BIG-BL palette (diverging)
    - Toggle through L1, L2, L3, L4 maps
    - Look for:
      * Early Visual (V1/V2): Should be BLUE (especially in L4)
      * Parietal (SPL/IPS): Should be RED (especially in L4)
    """)


if __name__ == "__main__":
    main()