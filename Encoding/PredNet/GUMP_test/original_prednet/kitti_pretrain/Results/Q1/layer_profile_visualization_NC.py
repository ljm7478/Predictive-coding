"""
Wen et al. 2018 Style Visualization
- Best Layer Map: Each vertex colored by which layer has highest beta
- ROI Layer Profile: Line plots showing beta across layers for each ROI
- With SEM error bars from individual subjects
- Option to use NC (Noise Ceiling) normalized beta
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================

GLASSER_LABEL_PATH = '/combinelab2/03_user/jungmin/02_data/17_MMP_label/mmp/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'

# --- Choose: Raw Beta or NC Normalized ---
USE_NC_NORMALIZED = True  # Set to False for raw beta
USE_NC_THRESHOLD = False

# Set to True to use log10-transformed encoding output directory
USE_LOG10 = True

# Base encoding directory
if USE_LOG10:
    BASE_ENCODING_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4'
else:
    BASE_ENCODING_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4'

# Raw beta paths (LORO)
BETA_DIR = BASE_ENCODING_DIR

# NC Normalized beta paths
# Option 1: No threshold (original)
NC_BETA_DIR = os.path.join(BASE_ENCODING_DIR, 'divergence_analysis', 'step1_normalized_beta', 'no_thresh_range_global_merged')

# NC file names
NC_AHAT_FILE = 'ahat_beta_norm_global.dscalar.nii'
NC_E_FILE = 'e_beta_norm_global.dscalar.nii'

# Output directory (changes based on NC or raw)
if USE_NC_THRESHOLD:
    SAVE_DIR = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'layer_profile_NC_ISC_thresh_0.1')
else:
    SAVE_DIR = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'layer_profile')

# GUMP subjects
SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

# ROI definitions - Pure Visual Processing Hierarchy + Visuomotor
VISUAL_HIERARCHY_SIMPLE = {
    'Early Visual': ['V1', 'V2', 'V3', 'V4'],      # Low-level visual
    'Mid-Dorsal': ['V3A', 'V3B'],                   # Dorsal stream
    'Motion': ['MT', 'MST'],                        # Motion processing
    'Object': ['LO1', 'LO2', 'LO3'],               # Lateral Occipital - Object recognition
    'Face': ['FFC'],                                # Fusiform Face Complex
    'Scene': ['PHA1', 'PHA2', 'PHA3'],             # Parahippocampal - Scene/Place
    'Attention': ['FEF', 'IPS1', 'PCV', '23c'],                   # Frontal Eye Field, Intraparietal
    'Visuomotor': ['AIP', 'LIPd', 'LIPv', 'MIP', 'VIP',  # Intraparietal areas
                   'PF', 'PFm', 'PFt',                # Action observation
                    '6r', '6v'],                      # Eye/Action related 
}

HIERARCHY_ORDER = ['Early Visual', 'Mid-Dorsal', 'Motion', 'Object', 'Face', 'Scene', 'Attention', 'Visuomotor']

N_LAYERS = 4

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_glasser_labels(dlabel_path):
    """Load Glasser parcellation labels"""
    img = nib.load(dlabel_path)
    labels = img.get_fdata().squeeze()
    header = img.header
    label_table = header.get_axis(0).label[0]
    label_dict = {}
    for idx, (name, rgba) in label_table.items():
        label_dict[idx] = name
    return labels, label_dict


def get_roi_indices(labels, label_dict, roi_names):
    """Get vertex indices for specified ROI names"""
    name_to_idx = {v: k for k, v in label_dict.items()}
    all_indices = []
    for roi_name in roi_names:
        for hemi_prefix in ['L_', 'R_']:
            full_name = "{}{}_ROI".format(hemi_prefix, roi_name)
            if full_name in name_to_idx:
                roi_idx = name_to_idx[full_name]
                vertex_indices = np.where(labels == roi_idx)[0]
                all_indices.extend(vertex_indices)
    return np.array(all_indices)


def load_beta_cifti(beta_path):
    """Load beta values from CIFTI file"""
    img = nib.load(beta_path)
    data = img.get_fdata()
    return data


def compute_best_layer_map(beta_data):
    """
    Compute which layer has the highest beta for each vertex
    Returns: array of layer indices (1-4) for each vertex
    """
    # beta_data shape: (n_layers, n_vertices)
    best_layer = np.argmax(beta_data, axis=0) + 1  # 1-indexed
    return best_layer


def extract_roi_layer_profile(beta_data, labels, label_dict, roi_hierarchy, hierarchy_order, n_layers=4):
    """
    Extract mean beta per layer for each ROI
    Returns: dict of {roi_name: [layer1_beta, layer2_beta, ...]}
    Handles shape mismatch between labels and beta_data
    """
    n_vertices_beta = beta_data.shape[-1]
    
    profiles = {}
    for roi_name in hierarchy_order:
        region_names = roi_hierarchy[roi_name]
        indices = get_roi_indices(labels, label_dict, region_names)
        
        # Filter indices to valid range
        indices = indices[indices < n_vertices_beta]
        
        if len(indices) > 0:
            layer_betas = []
            for layer_idx in range(n_layers):
                mean_beta = np.nanmean(beta_data[layer_idx, indices])
                layer_betas.append(mean_beta)
            profiles[roi_name] = layer_betas
        else:
            profiles[roi_name] = [np.nan] * n_layers
    return profiles


def extract_roi_layer_profile_with_sem(beta_dir, subjects, labels, label_dict, 
                                        roi_hierarchy, hierarchy_order, 
                                        signal_type='Ahat', n_layers=4,
                                        use_nc=False, nc_dir=None):
    """
    Extract mean beta and SEM per layer for each ROI across subjects
    
    Parameters:
    -----------
    use_nc : bool
        If True, use NC normalized beta files
    nc_dir : str
        Path to NC normalized beta directory
    
    Returns:
    --------
    profiles_mean : dict of {roi_name: [layer1_mean, layer2_mean, ...]}
    profiles_sem : dict of {roi_name: [layer1_sem, layer2_sem, ...]}
    """
    n_subjects = len(subjects)
    n_rois = len(hierarchy_order)
    
    # Initialize: (n_subjects, n_layers, n_rois)
    all_subject_data = np.zeros((n_subjects, n_layers, n_rois))
    
    for subj_idx, subj in enumerate(subjects):
        if use_nc and nc_dir is not None:
            # NC normalized paths
            if signal_type == 'Ahat':
                nc_file = 'ahat_beta_norm_global.dscalar.nii'
            else:
                nc_file = 'e_beta_norm_global.dscalar.nii'
            beta_path = os.path.join(nc_dir, nc_file)
            
            # NC files are group-level, not per-subject
            # Skip subject loop for NC (will handle below)
            break
        else:
            # Raw beta paths
            possible_paths = [
                os.path.join(beta_dir, signal_type, subj, 'beta_pc1_all_layers_LORO.npy'),
                os.path.join(beta_dir, signal_type, subj, 'beta_all_layers.npy'),
                os.path.join(beta_dir, signal_type, subj, '{}_beta.npy'.format(subj)),
            ]
            
            beta_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    beta_path = p
                    break
            
            if beta_path is None:
                print("    Warning: Beta file not found for {}".format(subj))
                all_subject_data[subj_idx, :, :] = np.nan
                continue
            
            # Load subject beta: shape (n_layers, n_vertices)
            beta_data = np.load(beta_path)
            
            # Extract ROI means
            for roi_idx, roi_name in enumerate(hierarchy_order):
                region_names = roi_hierarchy[roi_name]
                indices = get_roi_indices(labels, label_dict, region_names)
                
                if len(indices) > 0:
                    for layer_idx in range(n_layers):
                        mean_beta = np.nanmean(beta_data[layer_idx, indices])
                        all_subject_data[subj_idx, layer_idx, roi_idx] = mean_beta
                else:
                    all_subject_data[subj_idx, :, roi_idx] = np.nan
    
    # Handle NC normalized (group-level only, no SEM)
    if use_nc and nc_dir is not None:
        if signal_type == 'Ahat':
            # Try threshold version first, then no-threshold
            nc_path = os.path.join(nc_dir, 'ahat_beta_norm_thresh.dscalar.nii')
            if not os.path.exists(nc_path):
                nc_path = os.path.join(nc_dir, 'ahat_beta_norm_global.dscalar.nii')
        else:
            nc_path = os.path.join(nc_dir, 'e_beta_norm_thresh.dscalar.nii')
            if not os.path.exists(nc_path):
                nc_path = os.path.join(nc_dir, 'e_beta_norm_global.dscalar.nii')
        
        if os.path.exists(nc_path):
            beta_data = load_beta_cifti(nc_path)
            n_vertices_beta = beta_data.shape[-1]
            print("    Loaded NC normalized: {}".format(nc_path))
            print("    Shape: {}".format(beta_data.shape))
            
            profiles_mean = {}
            profiles_sem = {}
            
            for roi_idx, roi_name in enumerate(hierarchy_order):
                region_names = roi_hierarchy[roi_name]
                indices = get_roi_indices(labels, label_dict, region_names)
                
                # Filter indices to valid range (handle shape mismatch)
                indices = indices[indices < n_vertices_beta]
                
                if len(indices) > 0:
                    mean_vals = []
                    sem_vals = []
                    for layer_idx in range(n_layers):
                        roi_values = beta_data[layer_idx, indices]
                        mean_vals.append(np.nanmean(roi_values))
                        # SEM across vertices within ROI
                        sem_vals.append(np.nanstd(roi_values) / np.sqrt(np.sum(~np.isnan(roi_values))))
                    profiles_mean[roi_name] = mean_vals
                    profiles_sem[roi_name] = sem_vals
                else:
                    profiles_mean[roi_name] = [np.nan] * n_layers
                    profiles_sem[roi_name] = [np.nan] * n_layers
            
            return profiles_mean, profiles_sem
        else:
            print("    ERROR: NC file not found: {}".format(nc_path))
            return None, None
    
    # Compute mean and SEM across subjects (for raw beta)
    profiles_mean = {}
    profiles_sem = {}
    
    for roi_idx, roi_name in enumerate(hierarchy_order):
        roi_data = all_subject_data[:, :, roi_idx]  # (n_subjects, n_layers)
        
        # Mean across subjects
        mean_vals = np.nanmean(roi_data, axis=0)  # (n_layers,)
        
        # SEM = std / sqrt(n)
        std_vals = np.nanstd(roi_data, axis=0, ddof=1)
        n_valid = np.sum(~np.isnan(roi_data), axis=0)
        sem_vals = std_vals / np.sqrt(n_valid)
        
        profiles_mean[roi_name] = mean_vals.tolist()
        profiles_sem[roi_name] = sem_vals.tolist()
    
    return profiles_mean, profiles_sem


def save_best_layer_cifti(best_layer_map, template_path, output_path):
    """Save best layer map as CIFTI dscalar"""
    template_img = nib.load(template_path)
    template_header = template_img.header
    template_data = template_img.get_fdata()
    n_vertices_template = template_data.shape[-1]
    
    brain_model_axis = template_header.get_axis(1)
    
    from nibabel.cifti2 import Cifti2Image
    from nibabel.cifti2.cifti2_axes import ScalarAxis
    
    scalar_axis = ScalarAxis(['Best_Layer'])
    new_header = nib.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    
    # Handle shape mismatch
    if len(best_layer_map) != n_vertices_template:
        print("    Warning: best_layer_map ({}) != template ({})".format(len(best_layer_map), n_vertices_template))
        # Resize to match template
        if len(best_layer_map) > n_vertices_template:
            best_layer_map = best_layer_map[:n_vertices_template]
        else:
            # Pad with zeros
            padded = np.zeros(n_vertices_template, dtype=np.float32)
            padded[:len(best_layer_map)] = best_layer_map
            best_layer_map = padded
    
    # Reshape to (1, n_vertices)
    data = best_layer_map.reshape(1, -1).astype(np.float32)
    
    new_img = Cifti2Image(data, header=new_header)
    nib.save(new_img, output_path)
    print("    Saved: {}".format(output_path))


def save_roi_mask_cifti(labels, label_dict, roi_hierarchy, hierarchy_order, 
                        template_path, output_path):
    """
    Save ROI mask as CIFTI dscalar
    Each ROI gets a unique integer label
    Handles cortical-only vs full (cortical+subcortical) templates
    """
    # Load template to get correct shape
    template_img = nib.load(template_path)
    template_data = template_img.get_fdata()
    n_vertices_template = template_data.shape[-1]
    
    n_vertices_labels = len(labels)
    
    print("    Template vertices: {}".format(n_vertices_template))
    print("    Glasser labels vertices: {}".format(n_vertices_labels))
    
    # Handle shape mismatch (cortical-only vs full)
    if n_vertices_template != n_vertices_labels:
        print("    Shape mismatch: using first {} vertices".format(n_vertices_template))
        # Assume cortical vertices come first
        labels_subset = labels[:n_vertices_template]
    else:
        labels_subset = labels
    
    roi_mask = np.zeros(n_vertices_template, dtype=np.float32)
    
    for roi_idx, roi_name in enumerate(hierarchy_order, start=1):
        region_names = roi_hierarchy[roi_name]
        # Get indices from full labels, then filter to valid range
        indices = get_roi_indices(labels, label_dict, region_names)
        # Keep only indices within template range
        indices = indices[indices < n_vertices_template]
        if len(indices) > 0:
            roi_mask[indices] = roi_idx
            print("    {}: {} vertices (label={})".format(roi_name, len(indices), roi_idx))
    
    # Save as CIFTI
    template_header = template_img.header
    brain_model_axis = template_header.get_axis(1)
    
    from nibabel.cifti2 import Cifti2Image
    from nibabel.cifti2.cifti2_axes import ScalarAxis
    
    scalar_axis = ScalarAxis(['ROI_Mask'])
    new_header = nib.Cifti2Header.from_axes((scalar_axis, brain_model_axis))
    
    data = roi_mask.reshape(1, -1)
    new_img = Cifti2Image(data, header=new_header)
    nib.save(new_img, output_path)
    print("    Saved ROI mask: {}".format(output_path))
    
    return roi_mask


# ============================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================

def plot_roi_layer_profile(pred_profiles, error_profiles, hierarchy_order,
                          pred_sem=None, error_sem=None, save_path=None,
                          title_suffix=''):
    """
    Create line plots showing layer profile for each ROI
    Similar to Wen et al. 2018 Figure (c) bottom panels
    With optional SEM error bars
    """
    n_rois = len(hierarchy_order)
    fig, axes = plt.subplots(1, n_rois, figsize=(2.8*n_rois, 3.5), sharey=True)
    
    if n_rois == 1:
        axes = [axes]
    
    layers = np.arange(1, N_LAYERS + 1)
    
    # Colors - Prediction blue, Error red
    pred_color = '#3498DB'  # Blue for Prediction
    error_color = '#E74C3C'  # Red for Error
    
    # Calculate unified y-axis limits
    all_vals = []
    for roi_name in hierarchy_order:
        all_vals.extend(pred_profiles[roi_name])
        all_vals.extend(error_profiles[roi_name])
    
    if pred_sem is not None:
        for roi_name in hierarchy_order:
            for m, s in zip(pred_profiles[roi_name], pred_sem[roi_name]):
                all_vals.extend([m - s, m + s])
    if error_sem is not None:
        for roi_name in hierarchy_order:
            for m, s in zip(error_profiles[roi_name], error_sem[roi_name]):
                all_vals.extend([m - s, m + s])
    
    y_min = min(all_vals) * 0.9 if min(all_vals) > 0 else min(all_vals) * 1.1
    y_max = max(all_vals) * 1.1
    
    for idx, (ax, roi_name) in enumerate(zip(axes, hierarchy_order)):
        pred_betas = pred_profiles[roi_name]
        error_betas = error_profiles[roi_name]
        
        # Plot with error bars if SEM provided
        if pred_sem is not None and error_sem is not None:
            pred_sem_vals = pred_sem[roi_name]
            error_sem_vals = error_sem[roi_name]
            
            ax.errorbar(layers, pred_betas, yerr=pred_sem_vals, 
                       fmt='o-', color=pred_color, linewidth=2, 
                       markersize=8, label='Prediction', 
                       markerfacecolor='white', markeredgewidth=2,
                       capsize=4, capthick=1.5, elinewidth=1.5)
            ax.errorbar(layers, error_betas, yerr=error_sem_vals,
                       fmt='s-', color=error_color, linewidth=2, 
                       markersize=8, label='Error', 
                       markerfacecolor='white', markeredgewidth=2,
                       capsize=4, capthick=1.5, elinewidth=1.5)
        else:
            ax.plot(layers, pred_betas, 'o-', color=pred_color, linewidth=2, 
                    markersize=8, label='Prediction', markerfacecolor='white', markeredgewidth=2)
            ax.plot(layers, error_betas, 's-', color=error_color, linewidth=2, 
                    markersize=8, label='Error', markerfacecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('Layer', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Mean Beta', fontsize=12)
        ax.set_title(roi_name, fontsize=13, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(y_min, y_max)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend only on first plot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle('Layer Profile by ROI (GUMP){}'.format(title_suffix), fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(save_path))
    
    plt.close()


def plot_combined_layer_profile(pred_profiles, error_profiles, hierarchy_order, 
                                pred_sem=None, error_sem=None, save_path=None,
                                title_suffix=''):
    """
    Create combined line plot with all ROIs on same axes
    With optional SEM error bars
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = np.arange(1, N_LAYERS + 1)
    
    # Color palette for 8 ROIs
    colors = ['#2C3E50', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6', '#3498DB', '#1ABC9C', '#E91E63']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X']
    
    # Calculate unified y-axis limits
    all_pred_vals = []
    all_error_vals = []
    for roi_name in hierarchy_order:
        all_pred_vals.extend(pred_profiles[roi_name])
        all_error_vals.extend(error_profiles[roi_name])
    
    if pred_sem is not None:
        for roi_name in hierarchy_order:
            for i, (m, s) in enumerate(zip(pred_profiles[roi_name], pred_sem[roi_name])):
                all_pred_vals.extend([m - s, m + s])
    if error_sem is not None:
        for roi_name in hierarchy_order:
            for i, (m, s) in enumerate(zip(error_profiles[roi_name], error_sem[roi_name])):
                all_error_vals.extend([m - s, m + s])
    
    all_vals = all_pred_vals + all_error_vals
    y_min = min(all_vals) * 0.9 if min(all_vals) > 0 else min(all_vals) * 1.1
    y_max = max(all_vals) * 1.1
    
    # Prediction plot
    ax = axes[0]
    for idx, roi_name in enumerate(hierarchy_order):
        betas = pred_profiles[roi_name]
        
        if pred_sem is not None:
            sem = pred_sem[roi_name]
            ax.errorbar(layers, betas, yerr=sem, marker=markers[idx], color=colors[idx], 
                       linewidth=2.5, markersize=10, label=roi_name,
                       markerfacecolor='white', markeredgewidth=2,
                       capsize=4, capthick=2, elinewidth=1.5)
        else:
            ax.plot(layers, betas, marker=markers[idx], color=colors[idx], 
                   linewidth=2.5, markersize=10, label=roi_name,
                   markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Mean Beta', fontsize=12)
    ax.set_title('Prediction (Ahat)', fontsize=13, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xlim(0.5, N_LAYERS + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Error plot
    ax = axes[1]
    for idx, roi_name in enumerate(hierarchy_order):
        betas = error_profiles[roi_name]
        
        if error_sem is not None:
            sem = error_sem[roi_name]
            ax.errorbar(layers, betas, yerr=sem, marker=markers[idx], color=colors[idx], 
                       linewidth=2.5, markersize=10, label=roi_name,
                       markerfacecolor='white', markeredgewidth=2,
                       capsize=4, capthick=2, elinewidth=1.5)
        else:
            ax.plot(layers, betas, marker=markers[idx], color=colors[idx], 
                   linewidth=2.5, markersize=10, label=roi_name,
                   markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Mean Beta', fontsize=12)
    ax.set_title('Error (E)', fontsize=13, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xlim(0.5, N_LAYERS + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.suptitle('Layer Profile Across Visual Hierarchy (GUMP){}'.format(title_suffix), fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(save_path))
    
    plt.close()


def plot_best_layer_histogram(pred_best, error_best, labels, label_dict, 
                               roi_hierarchy, hierarchy_order, save_path=None):
    """
    Create histogram showing distribution of best layers per ROI
    Handles shape mismatch between labels and best_layer arrays
    """
    n_rois = len(hierarchy_order)
    fig, axes = plt.subplots(2, n_rois, figsize=(2.5*n_rois, 6))
    
    layer_colors = ['#3498DB', '#2ECC71', '#F1C40F', '#E74C3C']  # Blue, Green, Yellow, Red for layers 1-4
    
    n_vertices_pred = len(pred_best)
    n_vertices_error = len(error_best)
    
    for col_idx, roi_name in enumerate(hierarchy_order):
        region_names = roi_hierarchy[roi_name]
        indices = get_roi_indices(labels, label_dict, region_names)
        
        if len(indices) == 0:
            continue
        
        for row_idx, (best_layer, title, n_vertices) in enumerate([
            (pred_best, 'Prediction', n_vertices_pred), 
            (error_best, 'Error', n_vertices_error)]):
            ax = axes[row_idx, col_idx]
            
            # Filter indices to valid range
            valid_indices = indices[indices < n_vertices]
            
            if len(valid_indices) == 0:
                continue
                
            roi_best = best_layer[valid_indices]
            
            # Count per layer
            counts = [np.sum(roi_best == l) for l in range(1, N_LAYERS + 1)]
            percentages = [c / len(roi_best) * 100 for c in counts]
            
            bars = ax.bar(range(1, N_LAYERS + 1), percentages, color=layer_colors, 
                         edgecolor='black', linewidth=1)
            
            ax.set_xlabel('Layer', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('{}\n% Vertices'.format(title), fontsize=10)
            ax.set_title('{}'.format(roi_name) if row_idx == 0 else '', fontsize=11, fontweight='bold')
            ax.set_xticks(range(1, N_LAYERS + 1))
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                if pct > 5:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           '{:.0f}%'.format(pct), ha='center', fontsize=8)
    
    plt.suptitle('Best Layer Distribution by ROI', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(save_path))
    
    plt.close()


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("="*70)
    print(" Wen et al. 2018 Style Visualization")
    print(" Best Layer Map & ROI Layer Profile")
    if USE_NC_NORMALIZED:
        print(" >>> Using NC (Noise Ceiling) Normalized Beta <<<")
    else:
        print(" >>> Using Raw Beta <<<")
    print("="*70)
    
    # --- Step 1: Load Glasser Labels ---
    print("\n[1] Loading Glasser parcellation...")
    labels, label_dict = load_glasser_labels(GLASSER_LABEL_PATH)
    print("    Loaded {} labels, {} vertices".format(len(label_dict), len(labels)))
    
    # --- Step 2: Create output directory ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\n[2] Output directory: {}".format(SAVE_DIR))
    
    # --- Step 3: Load beta files ---
    print("\n[3] Loading beta files...")
    
    if USE_NC_NORMALIZED:
        # NC normalized paths - try threshold version first
        ahat_beta_path = os.path.join(NC_BETA_DIR, 'ahat_beta_norm_thresh.dscalar.nii')
        error_beta_path = os.path.join(NC_BETA_DIR, 'e_beta_norm_thresh.dscalar.nii')
        
        # Fallback to no-threshold version
        if not os.path.exists(ahat_beta_path):
            ahat_beta_path = os.path.join(NC_BETA_DIR, 'ahat_beta_norm_global.dscalar.nii')
        if not os.path.exists(error_beta_path):
            error_beta_path = os.path.join(NC_BETA_DIR, 'e_beta_norm_global.dscalar.nii')
            
        title_suffix = ' - NC Normalized'
    else:
        # Raw beta paths (LORO)
        ahat_beta_path = os.path.join(BETA_DIR, 'Ahat', 'group', 'group_mean_betas_LORO.dscalar.nii')
        error_beta_path = os.path.join(BETA_DIR, 'E', 'group', 'group_mean_betas_LORO.dscalar.nii')
        title_suffix = ''
    
    # Check file existence
    if os.path.exists(ahat_beta_path):
        print("    Found Ahat: {}".format(ahat_beta_path))
    else:
        print("    ERROR: Ahat beta file not found: {}".format(ahat_beta_path))
        exit(1)
        
    if os.path.exists(error_beta_path):
        print("    Found Error: {}".format(error_beta_path))
    else:
        print("    ERROR: Error beta file not found: {}".format(error_beta_path))
        exit(1)
    
    ahat_beta = load_beta_cifti(ahat_beta_path)
    error_beta = load_beta_cifti(error_beta_path)
    print("    Ahat shape: {}".format(ahat_beta.shape))
    print("    Error shape: {}".format(error_beta.shape))
    
    # --- Step 3.5: Create ROI mask ---
    print("\n[3.5] Creating ROI mask...")
    roi_mask = save_roi_mask_cifti(
        labels, label_dict, VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
        ahat_beta_path, os.path.join(SAVE_DIR, 'visual_hierarchy_ROI_mask.dscalar.nii')
    )
    
    # --- Step 4: Compute Best Layer Maps ---
    print("\n[4] Computing Best Layer maps...")
    
    pred_best_layer = compute_best_layer_map(ahat_beta)
    error_best_layer = compute_best_layer_map(error_beta)
    
    print("    Prediction best layer distribution: {}".format([np.sum(pred_best_layer == l) for l in range(1, N_LAYERS + 1)]))
    print("    Error best layer distribution: {}".format([np.sum(error_best_layer == l) for l in range(1, N_LAYERS + 1)]))
    
    # Save as CIFTI
    save_best_layer_cifti(pred_best_layer, ahat_beta_path, 
                          os.path.join(SAVE_DIR, 'Prediction_best_layer.dscalar.nii'))
    save_best_layer_cifti(error_best_layer, error_beta_path,
                          os.path.join(SAVE_DIR, 'Error_best_layer.dscalar.nii'))
    
    # --- Step 5: Extract ROI Layer Profiles ---
    print("\n[5] Extracting ROI layer profiles...")
    
    if USE_NC_NORMALIZED:
        # NC normalized - group level only
        pred_mean, pred_sem = extract_roi_layer_profile_with_sem(
            BETA_DIR, SUBJECTS, labels, label_dict,
            VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
            signal_type='Ahat', n_layers=N_LAYERS,
            use_nc=True, nc_dir=NC_BETA_DIR
        )
        error_mean, error_sem = extract_roi_layer_profile_with_sem(
            BETA_DIR, SUBJECTS, labels, label_dict,
            VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
            signal_type='E', n_layers=N_LAYERS,
            use_nc=True, nc_dir=NC_BETA_DIR
        )
    else:
        # Raw beta with subject-level SEM
        pred_mean, pred_sem = extract_roi_layer_profile_with_sem(
            BETA_DIR, SUBJECTS, labels, label_dict,
            VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
            signal_type='Ahat', n_layers=N_LAYERS
        )
        error_mean, error_sem = extract_roi_layer_profile_with_sem(
            BETA_DIR, SUBJECTS, labels, label_dict,
            VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
            signal_type='E', n_layers=N_LAYERS
        )
    
    # Fallback to group data
    if pred_mean is None:
        pred_mean = extract_roi_layer_profile(ahat_beta, labels, label_dict, 
                                               VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER)
    if error_mean is None:
        error_mean = extract_roi_layer_profile(error_beta, labels, label_dict,
                                                VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER)
    
    print("\n    Prediction profiles:")
    for roi in HIERARCHY_ORDER:
        means = pred_mean[roi]
        print("      {}: {}".format(roi, ['{:.4f}'.format(m) for m in means]))
    
    print("\n    Error profiles:")
    for roi in HIERARCHY_ORDER:
        means = error_mean[roi]
        print("      {}: {}".format(roi, ['{:.4f}'.format(m) for m in means]))
    
    # --- Step 6: Create Visualizations ---
    print("\n[6] Creating visualizations...")
    
    # Individual ROI profiles
    plot_roi_layer_profile(pred_mean, error_mean, HIERARCHY_ORDER,
                          pred_sem=pred_sem, error_sem=error_sem,
                          save_path=os.path.join(SAVE_DIR, 'layer_profile_by_roi.png'),
                          title_suffix=title_suffix)
    
    # Combined profiles
    plot_combined_layer_profile(pred_mean, error_mean, HIERARCHY_ORDER,
                               pred_sem=pred_sem, error_sem=error_sem,
                               save_path=os.path.join(SAVE_DIR, 'layer_profile_combined.png'),
                               title_suffix=title_suffix)
    
    # Best layer histogram
    plot_best_layer_histogram(pred_best_layer, error_best_layer, labels, label_dict,
                              VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
                              save_path=os.path.join(SAVE_DIR, 'best_layer_histogram.png'))
    
    # --- Step 7: Save numerical data ---
    print("\n[7] Saving numerical data...")
    
    np.save(os.path.join(SAVE_DIR, 'pred_best_layer.npy'), pred_best_layer)
    np.save(os.path.join(SAVE_DIR, 'error_best_layer.npy'), error_best_layer)
    np.save(os.path.join(SAVE_DIR, 'pred_profiles_mean.npy'), pred_mean)
    np.save(os.path.join(SAVE_DIR, 'error_profiles_mean.npy'), error_mean)
    if pred_sem is not None:
        np.save(os.path.join(SAVE_DIR, 'pred_profiles_sem.npy'), pred_sem)
    if error_sem is not None:
        np.save(os.path.join(SAVE_DIR, 'error_profiles_sem.npy'), error_sem)
    
    print("\n" + "="*70)
    print(" Done!")
    print("="*70)
    print("\nOutput files saved to: {}".format(SAVE_DIR))
    print("\nMode: {}".format('NC Normalized' if USE_NC_NORMALIZED else 'Raw Beta'))