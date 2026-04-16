# -*- coding: utf-8 -*-
"""
Renders .mp4 videos from PredNet's pixel prediction features stored in HDF5 (.h5) files.

This script takes the seg0_Ahat.h5 file, extracts the 'Ahat0' (pixel prediction)
dataset, and uses ffmpeg to create a video, allowing for a visual
assessment of the model's prediction quality.

Python 2.7 compatible version.
"""
from __future__ import print_function  # Python 2/3 compatible print

import os
import numpy as np
import subprocess as sp
import warnings
import h5py  # Added for HDF5 file support

# --- Configuration ---
# Base directory where the .h5 feature files are saved.
BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/longer-time_result/prednet_original/h5/pt_kitti_ft_Lall_nt150/indirect_10s_extrap/layer4/'
# Specify which video run to render.
RUN_TO_RENDER = 'seg0'
# Directory to save the rendered videos.
VIDEO_SAVE_DIR = os.path.join(BASE_RESULTS_DIR)

# Frames per second (FPS) for the output video.
VIDEO_FPS = 25
# --- End Configuration ---


def render_video_from_h5(h5_path, dataset_key, output_video_path, fps):
    """
    Renders an .mp4 video from a dataset within an .h5 file.
    Assumes data shape is (B, T, H, W, C) and renders the first batch (index 0).
    Requires ffmpeg to be installed and available in the system's PATH.
    """
    if not os.path.exists(h5_path):
        print("Warning: Cannot render video. File not found: {0}".format(h5_path))
        return

    print("Opening HDF5 file: {0} (loading key: {1})...".format(h5_path, dataset_key))

    try:
        with h5py.File(h5_path, 'r') as f:
            if dataset_key not in f:
                print("Error: Dataset key '{0}' not found in {1}".format(dataset_key, h5_path))
                return
            
            # Load data for the first batch item (B, T, H, W, C) -> (T, H, W, C)
            # This loads the data into memory.
            video_data = f[dataset_key][0] 

        print("Rendering video from {0} (key: {1})...".format(h5_path, dataset_key))
        
        # Convert data for ffmpeg: uint8 format, range [0, 255]
        # Assumes input data is float and in range [0, 1]
        if np.max(video_data) <= 1.0 and np.min(video_data) >= 0.0:
             video_data = (video_data * 255).astype(np.uint8)
        else:
            warnings.warn("Input data is not in the expected [0, 1] range. Clamping.")
            video_data = np.clip(video_data, 0, 1)
            video_data = (video_data * 255).astype(np.uint8)
        
        t, h, w, c = video_data.shape
        
        if c != 3:
            warnings.warn("Video rendering is only supported for 3-channel (RGB) data. Found {0} channels. Skipping.".format(c))
            return

        # Set up the ffmpeg command
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '{0}x{1}'.format(w, h),  # width x height
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # The input comes from a pipe
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'slow',
            '-crf', '18',
            output_video_path
        ]

        # Execute the ffmpeg command and pipe the video data to it
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        pipe.stdin.write(video_data.tostring())  # Use .tostring() for Python 2.7
        pipe.stdin.close()
        stderr_output = pipe.stderr.read()  # No decode needed in Python 2.7
        pipe.wait()

        if pipe.returncode == 0:
            print("Video successfully rendered to: {0}".format(output_video_path))
        else:
            print("Error: ffmpeg command failed.")
            print("ffmpeg error output:\n{0}".format(stderr_output))

    except (IOError, OSError) as e:
        print("Error: ffmpeg command failed. Is ffmpeg installed and in your PATH? Error: {0}".format(e))
    except Exception as e:
        print("An unexpected error occurred: {0}".format(e))


if __name__ == "__main__":
    print("Starting video rendering for run: {0}".format(RUN_TO_RENDER))
    if not os.path.exists(VIDEO_SAVE_DIR):
        os.makedirs(VIDEO_SAVE_DIR)
        print("Created save directory: {0}".format(VIDEO_SAVE_DIR))

    run_dir = os.path.join(BASE_RESULTS_DIR)
    if not os.path.exists(run_dir):
        print("Error: Run directory not found at {0}".format(run_dir))
        exit()

    # Render the predicted video (Ahat0) from the HDF5 file
    ahat0_h5_path = os.path.join(run_dir, 'seg0_Ahat.h5')
    ahat0_video_path = os.path.join(VIDEO_SAVE_DIR, 'predicted_video_Ahat0_{0}.mp4'.format(RUN_TO_RENDER))
    
    render_video_from_h5(ahat0_h5_path, 'Ahat0', ahat0_video_path, VIDEO_FPS)

    print("\nRendering complete. Video saved in: {0}".format(VIDEO_SAVE_DIR))