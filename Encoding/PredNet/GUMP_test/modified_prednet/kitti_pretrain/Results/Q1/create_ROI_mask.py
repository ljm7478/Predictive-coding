"""
Create ROI Mask for Visualization
- Single dscalar file with all ROIs
- Each ROI has unique integer value (1-7)
- Apply discrete palette for distinct colors in wb_view
"""

import numpy as np
import os
import nibabel as nib
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================

GLASSER_LABEL_PATH = '/combinelab2/03_user/jungmin/02_data/17_MMP_label/mmp/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'

# Template CIFTI (for brain axis)
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Output
OUTPUT_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4/Results/Q1'

# ROI definitions (including Visuomotor)
VISUAL_HIERARCHY_SIMPLE = {
    'Early Visual': ['V1', 'V2', 'V3', 'V4'],
    'Mid-Dorsal': ['V3A', 'V3B'],
    'Motion': ['MT', 'MST'],
    'Object': ['LO1', 'LO2', 'LO3'],
    'Face': ['FFC'],
    'Scene': ['PHA1', 'PHA2', 'PHA3'],
    'Attention': ['FEF', 'IPS1'],
    'Visuomotor': ['AIP', 'LIPd', 'LIPv', 'MIP', 'VIP',
                   '1', '2', '3a', '3b',
                   '6a', '6d', '6r', '6v'],
}

HIERARCHY_ORDER = ['Early Visual', 'Mid-Dorsal', 'Motion', 'Object', 'Face', 'Scene', 'Attention', 'Visuomotor']

# Distinct colors for each ROI (RGB 0-255)
ROI_COLORS = {
    'Early Visual': (31, 119, 180),    # Blue
    'Mid-Dorsal': (255, 127, 14),      # Orange
    'Motion': (44, 160, 44),           # Green
    'Object': (214, 39, 40),           # Red
    'Face': (148, 103, 189),           # Purple
    'Scene': (140, 86, 75),            # Brown
    'Attention': (227, 119, 194),      # Pink
    'Visuomotor': (127, 127, 127),     # Gray
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def create_roi_mask_dscalar(labels, label_dict, roi_hierarchy, hierarchy_order,
                            template_path, output_path):
    """
    Create ROI mask as dscalar.nii with all ROIs in one file
    Values: 0 = background, 1-7 = ROIs
    """
    # Load template
    template = nib.load(template_path)
    brain_axis = template.header.get_axis(1)
    n_vertices_template = brain_axis.name.shape[0]
    
    print("Template vertices: {}".format(n_vertices_template))
    print("Glasser labels vertices: {}".format(len(labels)))
    
    # Initialize mask
    roi_mask = np.zeros(n_vertices_template, dtype=np.float32)
    
    print("\nROI Assignment:")
    print("-" * 50)
    
    total_vertices = 0
    for roi_idx, roi_name in enumerate(hierarchy_order, start=1):
        region_names = roi_hierarchy[roi_name]
        indices = get_roi_indices(labels, label_dict, region_names)
        
        # Filter to valid range
        indices = indices[indices < n_vertices_template]
        
        if len(indices) > 0:
            roi_mask[indices] = roi_idx
            total_vertices += len(indices)
            print("  {} = {:15s} : {:5d} vertices ({})".format(
                roi_idx, roi_name, len(indices), ', '.join(region_names)))
    
    print("-" * 50)
    print("  Total ROI vertices: {}".format(total_vertices))
    print("  Background vertices: {}".format(n_vertices_template - total_vertices))
    
    # Create CIFTI
    scalar_axis = nib.cifti2.ScalarAxis(['ROI_Mask'])
    header = nib.Cifti2Header.from_axes((scalar_axis, brain_axis))
    
    data = roi_mask.reshape(1, -1)
    img = nib.Cifti2Image(data.astype(np.float32), header)
    nib.save(img, output_path)
    
    print("\nSaved: {}".format(output_path))
    
    return roi_mask


def apply_discrete_palette(cifti_path, hierarchy_order, roi_colors):
    """
    Apply discrete color palette using wb_command
    Makes each ROI a distinct color in wb_view
    """
    # Create palette file content
    # wb_command -cifti-palette uses specific format
    
    # For discrete labels, we use -palette-name with custom range
    # Or create a .scene file / label table
    
    # Simple approach: use MODE_USER_SCALE with specific range
    n_rois = len(hierarchy_order)
    
    try:
        subprocess.run([
            "wb_command", "-cifti-palette", cifti_path,
            "MODE_USER_SCALE", cifti_path,
            "-pos-user", "0.5", str(n_rois + 0.5),
            "-palette-name", "videen_style",  # Good for discrete values
            "-disp-pos", "true",
            "-disp-neg", "false",
            "-disp-zero", "false"
        ], check=True)
        print("Applied discrete palette")
    except Exception as e:
        print("Warning: Could not apply palette: {}".format(e))


def create_roi_mask_dlabel(labels, label_dict, roi_hierarchy, hierarchy_order,
                           roi_colors, template_path, output_path):
    """
    Create ROI mask as dlabel.nii (discrete label file)
    This is better for categorical data - shows ROI names in wb_view
    """
    # Load template
    template = nib.load(template_path)
    brain_axis = template.header.get_axis(1)
    n_vertices_template = brain_axis.name.shape[0]
    
    # Initialize mask
    roi_mask = np.zeros(n_vertices_template, dtype=np.int32)
    
    print("\nCreating dlabel file...")
    
    for roi_idx, roi_name in enumerate(hierarchy_order, start=1):
        region_names = roi_hierarchy[roi_name]
        indices = get_roi_indices(labels, label_dict, region_names)
        indices = indices[indices < n_vertices_template]
        
        if len(indices) > 0:
            roi_mask[indices] = roi_idx
    
    # Create label table
    label_table = {
        0: ('Background', (0.5, 0.5, 0.5, 0.0)),  # Gray, transparent
    }
    
    for roi_idx, roi_name in enumerate(hierarchy_order, start=1):
        r, g, b = roi_colors[roi_name]
        # Convert 0-255 to 0-1
        label_table[roi_idx] = (roi_name, (r/255, g/255, b/255, 1.0))
    
    # Create LabelAxis
    label_axis = nib.cifti2.LabelAxis(['ROI_Mask'], {0: label_table})
    
    # Create header
    header = nib.Cifti2Header.from_axes((label_axis, brain_axis))
    
    # Create image
    data = roi_mask.reshape(1, -1)
    img = nib.Cifti2Image(data.astype(np.float32), header)
    
    # Save
    nib.save(img, output_path)
    print("Saved dlabel: {}".format(output_path))
    
    return roi_mask


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("="*60)
    print(" Creating ROI Mask for Visualization")
    print("="*60)
    
    # Load Glasser labels
    print("\n[1] Loading Glasser parcellation...")
    labels, label_dict = load_glasser_labels(GLASSER_LABEL_PATH)
    print("    Loaded {} labels".format(len(label_dict)))
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create dscalar version (numeric values)
    print("\n[2] Creating dscalar ROI mask...")
    dscalar_path = os.path.join(OUTPUT_DIR, 'ROI_mask_all_regions.dscalar.nii')
    roi_mask = create_roi_mask_dscalar(
        labels, label_dict, 
        VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
        CIFTI_TEMPLATE, dscalar_path
    )
    
    # Apply palette
    print("\n[3] Applying discrete palette...")
    apply_discrete_palette(dscalar_path, HIERARCHY_ORDER, ROI_COLORS)
    
    # Create dlabel version (with ROI names visible in wb_view)
    print("\n[4] Creating dlabel ROI mask (with labels)...")
    dlabel_path = os.path.join(OUTPUT_DIR, 'ROI_mask_all_regions.dlabel.nii')
    try:
        create_roi_mask_dlabel(
            labels, label_dict,
            VISUAL_HIERARCHY_SIMPLE, HIERARCHY_ORDER,
            ROI_COLORS, CIFTI_TEMPLATE, dlabel_path
        )
    except Exception as e:
        print("    Warning: dlabel creation failed: {}".format(e))
        print("    (dscalar version still available)")
    
    # Print legend
    print("\n" + "="*60)
    print(" ROI LEGEND (for wb_view)")
    print("="*60)
    print("\n Value | ROI Name       | Regions")
    print("-"*60)
    for roi_idx, roi_name in enumerate(HIERARCHY_ORDER, start=1):
        regions = VISUAL_HIERARCHY_SIMPLE[roi_name]
        print("   {}   | {:14s} | {}".format(roi_idx, roi_name, ', '.join(regions)))
    print("-"*60)
    print("\n   0   | Background     | (not in any ROI)")
    
    print("\n" + "="*60)
    print(" Output Files:")
    print("="*60)
    print("  1. {}".format(dscalar_path))
    print("     -> Open in wb_view, values 1-7 = ROIs")
    print("\n  2. {}".format(dlabel_path))
    print("     -> Open in wb_view, ROI names visible!")
    
    print("\n" + "="*60)
    print(" Done!")
    print("="*60)