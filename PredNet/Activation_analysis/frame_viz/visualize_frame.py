# -*- coding: utf-8 -*-
"""
Visualizes Prediction (Ahat) and Error (E) frames from PredNet models.
Creates side-by-side comparisons across layers to help understand what
the predicted frames and error frames look like compared to original input.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

# --- Configuration ---
# --- Model 1: Modified PredNet (CNN-like) ---
MODIFIED_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/CNN_like/h5/pt_kitti_ft_Lall_nt150/layer4/'
MODIFIED_RUN = 'seg0'
MODIFIED_LABEL = 'Modified PredNet (CNN-like)'

# --- Model 2: Original PredNet ---
ORIGINAL_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4'
ORIGINAL_RUN = 'seg0'
ORIGINAL_LABEL = 'Original PredNet'

# --- Original Input Frames ---
ORIGINAL_INPUT_PATH = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/longer-time_result/indirect_1s_extrap/npy/pt_kitti_ft_Lall_nt150/layer4/run1/original_video.npy'

# --- Output Settings ---
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_trend/test_GUMP/frame_visualization/'
NUM_LAYERS = 4
FRAME_INDEX = 1700  # Which frame to visualize (adjust based on your sequence length)
# --- End Configuration ---


def load_original_video(path):
    """
    Loads original video frames from .npy or .h5 file.
    Returns array of shape (time, channels, height, width) or (time, height, width, channels)
    """
    if path is None or not os.path.exists(path):
        print("  Original video not found: {}".format(path))
        return None
    
    print("  Loading original video from: {}".format(path))
    
    if path.endswith('.npy'):
        data = np.load(path)
    elif path.endswith('.h5'):
        with h5py.File(path, 'r') as hf:
            # Try common key names
            for key in ['video', 'frames', 'A', 'A0', 'data']:
                if key in hf:
                    data = hf[key][:]
                    break
            else:
                data = hf[list(hf.keys())[0]][:]
    else:
        print("  Unsupported format: {}".format(path))
        return None
    
    print("    Original video shape: {}".format(data.shape))
    return data


def get_original_frame_for_display(data, frame_idx=0):
    """
    Extracts a frame from the original video for display.
    Handles various input shapes and normalizes for display.
    """
    if data is None:
        return None
    
    # Handle batch dimension if present
    if data.ndim == 5:
        data = data[0]
    
    if frame_idx >= data.shape[0]:
        frame_idx = data.shape[0] - 1
        print("  Warning: frame_idx adjusted to {}".format(frame_idx))
    
    frame = data[frame_idx]
    
    # Determine if channels-first or channels-last
    if frame.ndim == 3:
        if frame.shape[0] in [1, 3]:  # Channels first
            if frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)  # -> (H, W, C)
            else:
                frame = frame[0]  # Grayscale -> (H, W)
        elif frame.shape[-1] in [1, 3]:  # Channels last
            if frame.shape[-1] == 1:
                frame = frame[:, :, 0]
    
    # Normalize to [0, 1] for display
    if frame.max() > 1.0:
        frame = frame / 255.0
    frame = np.clip(frame, 0, 1)
    
    return frame


def load_frames_from_h5(base_dir, run_id, num_layers):
    """
    Loads Ahat and E frames from HDF5 files.
    Returns dict with keys like 'Ahat0', 'Ahat1', ..., 'E0', 'E1', ...
    """
    frames = {}
    
    # Load Ahat (Prediction) frames
    ahat_file = os.path.join(base_dir, '{}_Ahat.h5'.format(run_id))
    if os.path.exists(ahat_file):
        print("  Loading Ahat from: {}".format(ahat_file))
        with h5py.File(ahat_file, 'r') as hf:
            print("  Available keys: {}".format(list(hf.keys())))
            for l in range(num_layers):
                layer_key = 'Ahat{}'.format(l)
                if layer_key in hf:
                    data = hf[layer_key][:]
                    frames[layer_key] = data
                    print("    {}: shape {}".format(layer_key, data.shape))
                elif '/' + layer_key in hf:
                    data = hf['/' + layer_key][:]
                    frames[layer_key] = data
                    print("    {}: shape {}".format(layer_key, data.shape))
    else:
        print("  Warning: File not found: {}".format(ahat_file))
    
    # Load E (Error) frames
    e_file = os.path.join(base_dir, '{}_E.h5'.format(run_id))
    if os.path.exists(e_file):
        print("  Loading E from: {}".format(e_file))
        with h5py.File(e_file, 'r') as hf:
            print("  Available keys: {}".format(list(hf.keys())))
            for l in range(num_layers):
                layer_key = 'E{}'.format(l)
                if layer_key in hf:
                    data = hf[layer_key][:]
                    frames[layer_key] = data
                    print("    {}: shape {}".format(layer_key, data.shape))
                elif '/' + layer_key in hf:
                    data = hf['/' + layer_key][:]
                    frames[layer_key] = data
                    print("    {}: shape {}".format(layer_key, data.shape))
    else:
        print("  Warning: File not found: {}".format(e_file))
    
    return frames


def get_frame_for_display(data, frame_idx=0, channel_mode='mean'):
    """
    Extracts a 2D image from the activation tensor for display.
    
    Args:
        data: 4D (time, channels, height, width) or 5D (batch, time, channels, height, width)
        frame_idx: Which time frame to extract
        channel_mode: 'mean' to average channels, 'rgb' to take first 3, 'first' for first channel
    
    Returns:
        2D or 3D numpy array suitable for imshow
    """
    # Handle 5D case (batch, time, channels, height, width)
    if data.ndim == 5:
        data = data[0]  # Take first batch
    
    # Now data is (time, channels, height, width)
    if frame_idx >= data.shape[0]:
        frame_idx = data.shape[0] - 1
        print("  Warning: frame_idx adjusted to {}".format(frame_idx))
    
    frame = data[frame_idx]  # (channels, height, width)
    
    if channel_mode == 'mean':
        return np.mean(frame, axis=0)
    elif channel_mode == 'rgb' and frame.shape[0] >= 3:
        rgb = frame[:3].transpose(1, 2, 0)  # (height, width, 3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb
    elif channel_mode == 'first':
        return frame[0]
    else:
        return np.mean(frame, axis=0)


def plot_original_vs_prediction_error(original_video, frames_dict, model_label, 
                                       save_path, frame_idx=50):
    """
    KEY VISUALIZATION: Shows Original Input | Prediction (Ahat0) | Error (E0) side by side.
    This helps viewers understand what the model is predicting vs ground truth.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('{} - Frame {} Comparison'.format(model_label, frame_idx), 
                 fontsize=14, fontweight='bold')
    
    # Column 0: Original Input
    ax = axes[0]
    if original_video is not None:
        orig_frame = get_original_frame_for_display(original_video, frame_idx)
        if orig_frame is not None:
            if orig_frame.ndim == 2:
                ax.imshow(orig_frame, cmap='gray')
            else:
                ax.imshow(orig_frame)
    ax.set_title('Original Input (Ground Truth)', fontsize=12)
    ax.axis('off')
    
    # Column 1: Prediction (Ahat0)
    ax = axes[1]
    if 'Ahat0' in frames_dict:
        pred_frame = get_frame_for_display(frames_dict['Ahat0'], frame_idx, 'rgb')
        if pred_frame.ndim == 2:
            ax.imshow(pred_frame, cmap='gray')
        else:
            ax.imshow(pred_frame)
    ax.set_title('Prediction (Â₀)', fontsize=12)
    ax.axis('off')
    
    # Column 2: Error (E0)
    ax = axes[2]
    if 'E0' in frames_dict:
        # For error, show mean across channels with diverging colormap
        err_frame = get_frame_for_display(frames_dict['E0'], frame_idx, 'mean')
        vmax = np.abs(err_frame).max()
        im = ax.imshow(err_frame, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Prediction Error (E₀)', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_dual_model_comparison_with_original(original_video, frames_original, frames_modified,
                                              save_path, frame_idx=50):
    """
    Comprehensive comparison: Original vs both PredNet architectures at L0.
    Layout: 2 rows x 3 cols
    Row 0: Original Input | Original PredNet Ahat0 | Modified PredNet Ahat0
    Row 1: (empty or diff) | Original PredNet E0   | Modified PredNet E0
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PredNet Architecture Comparison - Frame {}'.format(frame_idx), 
                 fontsize=14, fontweight='bold')
    
    # Row 0, Col 0: Original Input
    ax = axes[0, 0]
    if original_video is not None:
        orig_frame = get_original_frame_for_display(original_video, frame_idx)
        if orig_frame is not None:
            if orig_frame.ndim == 2:
                ax.imshow(orig_frame, cmap='gray')
            else:
                ax.imshow(orig_frame)
    ax.set_title('Original Input', fontsize=11)
    ax.axis('off')
    
    # Row 0, Col 1: Original PredNet Ahat0
    ax = axes[0, 1]
    if 'Ahat0' in frames_original:
        pred = get_frame_for_display(frames_original['Ahat0'], frame_idx, 'rgb')
        if pred.ndim == 2:
            ax.imshow(pred, cmap='gray')
        else:
            ax.imshow(pred)
    ax.set_title('Original PredNet - Prediction (Â₀)', fontsize=11)
    ax.axis('off')
    
    # Row 0, Col 2: Modified PredNet Ahat0
    ax = axes[0, 2]
    if 'Ahat0' in frames_modified:
        pred = get_frame_for_display(frames_modified['Ahat0'], frame_idx, 'rgb')
        if pred.ndim == 2:
            ax.imshow(pred, cmap='gray')
        else:
            ax.imshow(pred)
    ax.set_title('Modified PredNet - Prediction (Â₀)', fontsize=11)
    ax.axis('off')
    
    # Row 1, Col 0: Difference between original and predictions (optional)
    ax = axes[1, 0]
    ax.text(0.5, 0.5, 'Prediction\nError Maps\n→', ha='center', va='center', 
            fontsize=14, transform=ax.transAxes)
    ax.axis('off')
    
    # Row 1, Col 1: Original PredNet E0
    ax = axes[1, 1]
    if 'E0' in frames_original:
        err = get_frame_for_display(frames_original['E0'], frame_idx, 'mean')
        vmax = np.abs(err).max()
        im = ax.imshow(err, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Original PredNet - Error (E₀)', fontsize=11)
    ax.axis('off')
    
    # Row 1, Col 2: Modified PredNet E0
    ax = axes[1, 2]
    if 'E0' in frames_modified:
        err = get_frame_for_display(frames_modified['E0'], frame_idx, 'mean')
        vmax = np.abs(err).max()
        im = ax.imshow(err, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Modified PredNet - Error (E₀)', fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_temporal_with_original(original_video, frames_dict, model_label, save_path,
                                 start_frame=0, num_frames=6):
    """
    Shows temporal sequence: Original vs Prediction vs Error over time.
    Great for seeing how prediction quality evolves.
    """
    fig, axes = plt.subplots(3, num_frames, figsize=(3*num_frames, 9))
    fig.suptitle('{} - Temporal Sequence from Frame {}'.format(model_label, start_frame), 
                 fontsize=14, fontweight='bold')
    
    for i, t in enumerate(range(start_frame, start_frame + num_frames)):
        # Row 0: Original
        ax = axes[0, i]
        if original_video is not None:
            orig = get_original_frame_for_display(original_video, t)
            if orig is not None:
                if orig.ndim == 2:
                    ax.imshow(orig, cmap='gray')
                else:
                    ax.imshow(orig)
        ax.set_title('t={}'.format(t), fontsize=10)
        if i == 0:
            ax.set_ylabel('Original', fontsize=11)
        ax.axis('off')
        
        # Row 1: Prediction (Ahat0)
        ax = axes[1, i]
        if 'Ahat0' in frames_dict:
            pred = get_frame_for_display(frames_dict['Ahat0'], t, 'rgb')
            if pred.ndim == 2:
                ax.imshow(pred, cmap='gray')
            else:
                ax.imshow(pred)
        if i == 0:
            ax.set_ylabel('Prediction (Â₀)', fontsize=11)
        ax.axis('off')
        
        # Row 2: Error (E0)
        ax = axes[2, i]
        if 'E0' in frames_dict:
            err = get_frame_for_display(frames_dict['E0'], t, 'mean')
            vmax = np.abs(err).max()
            ax.imshow(err, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Error (E₀)', fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_layer_comparison(frames_dict, model_label, save_path, frame_idx=50, num_layers=4):
    """
    Creates a grid showing Ahat and E frames at each layer for one model.
    """
    fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
    fig.suptitle('{} - Frame {}'.format(model_label, frame_idx), fontsize=14, fontweight='bold')
    
    # Row 0: Prediction (Ahat)
    for l in range(num_layers):
        ax = axes[0, l]
        key = 'Ahat{}'.format(l)
        if key in frames_dict:
            data = frames_dict[key]
            if l == 0:
                img = get_frame_for_display(data, frame_idx, channel_mode='rgb')
            else:
                img = get_frame_for_display(data, frame_idx, channel_mode='mean')
            
            if img.ndim == 2:
                im = ax.imshow(img, cmap='viridis')
            else:
                im = ax.imshow(img)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Prediction (Â) L{}'.format(l))
        ax.axis('off')
    
    # Row 1: Error (E)
    for l in range(num_layers):
        ax = axes[1, l]
        key = 'E{}'.format(l)
        if key in frames_dict:
            data = frames_dict[key]
            img = get_frame_for_display(data, frame_idx, channel_mode='mean')
            vmax = np.abs(img).max()
            im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Error (E) L{}'.format(l))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_model_comparison_single_layer(frames_original, frames_modified, layer_idx, 
                                        save_path, frame_idx=50):
    """
    Compares Original vs Modified PredNet for a single layer.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Layer {} Comparison - Frame {}'.format(layer_idx, frame_idx), 
                 fontsize=14, fontweight='bold')
    
    titles = [
        ['Original PredNet - Prediction (Â)', 'Modified PredNet - Prediction (Â)'],
        ['Original PredNet - Error (E)', 'Modified PredNet - Error (E)']
    ]
    
    data_pairs = [
        [frames_original.get('Ahat{}'.format(layer_idx)), 
         frames_modified.get('Ahat{}'.format(layer_idx))],
        [frames_original.get('E{}'.format(layer_idx)), 
         frames_modified.get('E{}'.format(layer_idx))]
    ]
    
    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            data = data_pairs[row][col]
            
            if data is not None:
                if layer_idx == 0 and row == 0:
                    img = get_frame_for_display(data, frame_idx, channel_mode='rgb')
                else:
                    img = get_frame_for_display(data, frame_idx, channel_mode='mean')
                
                if row == 1:
                    vmax = np.abs(img).max()
                    im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                elif img.ndim == 2:
                    im = ax.imshow(img, cmap='viridis')
                else:
                    im = ax.imshow(img)
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', 
                       transform=ax.transAxes)
            
            ax.set_title(titles[row][col])
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_comprehensive_comparison(frames_original, frames_modified, save_path, 
                                   frame_idx=50, num_layers=4):
    """
    Creates a comprehensive figure showing all layers for both models.
    """
    fig, axes = plt.subplots(num_layers, 4, figsize=(16, 4*num_layers))
    fig.suptitle('PredNet Frame Comparison - Frame {}'.format(frame_idx), 
                 fontsize=16, fontweight='bold', y=1.02)
    
    col_titles = ['Original Prediction (Â)', 'Original Error (E)', 
                  'Modified Prediction (Â)', 'Modified Error (E)']
    
    for l in range(num_layers):
        # Original Ahat
        ax = axes[l, 0]
        key = 'Ahat{}'.format(l)
        if key in frames_original:
            if l == 0:
                img = get_frame_for_display(frames_original[key], frame_idx, 'rgb')
            else:
                img = get_frame_for_display(frames_original[key], frame_idx, 'mean')
            if img.ndim == 2:
                im = ax.imshow(img, cmap='viridis')
            else:
                im = ax.imshow(img)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('Layer {}'.format(l), fontsize=12, fontweight='bold')
        if l == 0:
            ax.set_title(col_titles[0])
        ax.axis('off')
        
        # Original E
        ax = axes[l, 1]
        key = 'E{}'.format(l)
        if key in frames_original:
            img = get_frame_for_display(frames_original[key], frame_idx, 'mean')
            vmax = np.abs(img).max()
            im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if l == 0:
            ax.set_title(col_titles[1])
        ax.axis('off')
        
        # Modified Ahat
        ax = axes[l, 2]
        key = 'Ahat{}'.format(l)
        if key in frames_modified:
            if l == 0:
                img = get_frame_for_display(frames_modified[key], frame_idx, 'rgb')
            else:
                img = get_frame_for_display(frames_modified[key], frame_idx, 'mean')
            if img.ndim == 2:
                im = ax.imshow(img, cmap='viridis')
            else:
                im = ax.imshow(img)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if l == 0:
            ax.set_title(col_titles[2])
        ax.axis('off')
        
        # Modified E
        ax = axes[l, 3]
        key = 'E{}'.format(l)
        if key in frames_modified:
            img = get_frame_for_display(frames_modified[key], frame_idx, 'mean')
            vmax = np.abs(img).max()
            im = ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if l == 0:
            ax.set_title(col_titles[3])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


def plot_temporal_sequence(frames_dict, model_label, save_path, 
                           layer_idx=0, start_frame=0, num_frames=8):
    """
    Shows temporal evolution of predictions and errors.
    """
    fig, axes = plt.subplots(2, num_frames, figsize=(3*num_frames, 6))
    fig.suptitle('{} - Layer {} Temporal Sequence'.format(model_label, layer_idx), 
                 fontsize=14, fontweight='bold')
    
    ahat_key = 'Ahat{}'.format(layer_idx)
    e_key = 'E{}'.format(layer_idx)
    
    for i, t in enumerate(range(start_frame, start_frame + num_frames)):
        # Ahat
        ax = axes[0, i]
        if ahat_key in frames_dict:
            if layer_idx == 0:
                img = get_frame_for_display(frames_dict[ahat_key], t, 'rgb')
            else:
                img = get_frame_for_display(frames_dict[ahat_key], t, 'mean')
            if img.ndim == 2:
                ax.imshow(img, cmap='viridis')
            else:
                ax.imshow(img)
        ax.set_title('t={}'.format(t))
        if i == 0:
            ax.set_ylabel('Prediction (Â)', fontsize=10)
        ax.axis('off')
        
        # E
        ax = axes[1, i]
        if e_key in frames_dict:
            img = get_frame_for_display(frames_dict[e_key], t, 'mean')
            vmax = np.abs(img).max()
            ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if i == 0:
            ax.set_ylabel('Error (E)', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: {}".format(save_path))


if __name__ == "__main__":
    print("=" * 60)
    print("PredNet Frame Visualization")
    print("=" * 60)
    
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        os.makedirs(ANALYSIS_SAVE_DIR)
    
    # --- Load original video ---
    print("\nLoading original video...")
    original_video = load_original_video(ORIGINAL_INPUT_PATH)
    
    # --- Load frames from both models ---
    print("\nLoading frames from Original PredNet...")
    frames_original = load_frames_from_h5(ORIGINAL_RESULTS_DIR, ORIGINAL_RUN, NUM_LAYERS)
    
    print("\nLoading frames from Modified PredNet...")
    frames_modified = load_frames_from_h5(MODIFIED_RESULTS_DIR, MODIFIED_RUN, NUM_LAYERS)
    
    # --- Generate visualizations ---
    
    # NEW: Key comparison with original input
    if original_video is not None:
        # Single model vs original
        if frames_original:
            plot_original_vs_prediction_error(
                original_video, frames_original, ORIGINAL_LABEL,
                os.path.join(ANALYSIS_SAVE_DIR, 'original_prednet_vs_input.png'),
                frame_idx=FRAME_INDEX
            )
        
        if frames_modified:
            plot_original_vs_prediction_error(
                original_video, frames_modified, MODIFIED_LABEL,
                os.path.join(ANALYSIS_SAVE_DIR, 'modified_prednet_vs_input.png'),
                frame_idx=FRAME_INDEX
            )
        
        # Dual model comparison with original
        if frames_original and frames_modified:
            plot_dual_model_comparison_with_original(
                original_video, frames_original, frames_modified,
                os.path.join(ANALYSIS_SAVE_DIR, 'both_models_vs_input.png'),
                frame_idx=FRAME_INDEX
            )
        
        # Temporal sequence with original
        if frames_original:
            plot_temporal_with_original(
                original_video, frames_original, ORIGINAL_LABEL,
                os.path.join(ANALYSIS_SAVE_DIR, 'original_prednet_temporal_with_input.png'),
                start_frame=FRAME_INDEX, num_frames=6
            )
        
        if frames_modified:
            plot_temporal_with_original(
                original_video, frames_modified, MODIFIED_LABEL,
                os.path.join(ANALYSIS_SAVE_DIR, 'modified_prednet_temporal_with_input.png'),
                start_frame=FRAME_INDEX, num_frames=6
            )
    
    # Individual model layer comparison
    if frames_original:
        plot_layer_comparison(
            frames_original, ORIGINAL_LABEL,
            os.path.join(ANALYSIS_SAVE_DIR, 'original_prednet_all_layers.png'),
            frame_idx=FRAME_INDEX, num_layers=NUM_LAYERS
        )
    
    if frames_modified:
        plot_layer_comparison(
            frames_modified, MODIFIED_LABEL,
            os.path.join(ANALYSIS_SAVE_DIR, 'modified_prednet_all_layers.png'),
            frame_idx=FRAME_INDEX, num_layers=NUM_LAYERS
        )
    
    # Comprehensive side-by-side comparison
    if frames_original and frames_modified:
        plot_comprehensive_comparison(
            frames_original, frames_modified,
            os.path.join(ANALYSIS_SAVE_DIR, 'comprehensive_comparison.png'),
            frame_idx=FRAME_INDEX, num_layers=NUM_LAYERS
        )
        
        # Per-layer comparisons
        for layer in [0, 3]:
            plot_model_comparison_single_layer(
                frames_original, frames_modified, layer,
                os.path.join(ANALYSIS_SAVE_DIR, 'comparison_layer{}.png'.format(layer)),
                frame_idx=FRAME_INDEX
            )
    
    # Temporal sequences (without original)
    if frames_original:
        plot_temporal_sequence(
            frames_original, ORIGINAL_LABEL,
            os.path.join(ANALYSIS_SAVE_DIR, 'original_temporal_L0.png'),
            layer_idx=0, start_frame=FRAME_INDEX, num_frames=8
        )
    
    if frames_modified:
        plot_temporal_sequence(
            frames_modified, MODIFIED_LABEL,
            os.path.join(ANALYSIS_SAVE_DIR, 'modified_temporal_L0.png'),
            layer_idx=0, start_frame=FRAME_INDEX, num_frames=8
        )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("Output directory: {}".format(ANALYSIS_SAVE_DIR))
    print("=" * 60)