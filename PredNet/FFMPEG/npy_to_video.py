"""
Renders .mp4 videos from PredNet's pixel prediction features (.npy files).

This script takes the Ahat0 (pixel prediction) and original_video .npy files
and uses ffmpeg to create videos, allowing for a direct visual comparison
of the model's prediction quality.
"""
import os
import numpy as np
import subprocess as sp
import warnings

# --- Configuration ---
# Base directory where the .npy feature files are saved.
# BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_concat6movies_ft_Lall_nt150/layer4/'
BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/CIT_500SUM_pretrained_Lall_nt150_finetuned/layer4/'

# Specify which video run to render.
RUN_TO_RENDER = 'run1'
# Directory to save the rendered videos.
VIDEO_SAVE_DIR = os.path.join(BASE_RESULTS_DIR, 'rendered_videos')

# Frames per second (FPS) for the output video.
VIDEO_FPS = 25
# --- End Configuration ---


def render_video_from_npy(npy_path, output_video_path, fps):
    """
    Renders an .mp4 video from a .npy file (B, T, H, W, C).
    Requires ffmpeg to be installed and available in the system's PATH.
    """
    if not os.path.exists(npy_path):
        print("Warning: Cannot render video. File not found: {}".format(npy_path))
        return

    print("Rendering video from {}...".format(npy_path))
    
    # Load data (B, T, H, W, C) in float32 format, range [0, 1]
    video_data = np.load(npy_path)[0]
    
    # Convert data for ffmpeg: uint8 format, range [0, 255]
    video_data = (video_data * 255).astype(np.uint8)
    
    t, h, w, c = video_data.shape
    
    if c != 3:
        warnings.warn("Video rendering is only supported for 3-channel (RGB) data. Skipping.")
        return

    # Set up the ffmpeg command
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '{}x{}'.format(w, h),  # width x height
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',  # The input comes from a pipe
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',
        '-crf', '18',
        output_video_path
    ]

    try:
        # Execute the ffmpeg command and pipe the video data to it
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        pipe.stdin.write(video_data.tostring())
        pipe.stdin.close()
        pipe.wait()
        print("Video successfully rendered to: {}".format(output_video_path))
    except (IOError, OSError) as e:
        print("Error: ffmpeg command failed. Is ffmpeg installed and in your PATH?")
        print("ffmpeg error output:\n{}".format(pipe.stderr.read()))


if __name__ == "__main__":
    print("Starting video rendering for run: {}".format(RUN_TO_RENDER))
    if not os.path.exists(VIDEO_SAVE_DIR):
        os.makedirs(VIDEO_SAVE_DIR)

    run_dir = os.path.join(BASE_RESULTS_DIR, RUN_TO_RENDER)
    if not os.path.exists(run_dir):
        print("Error: Run directory not found at {}".format(run_dir))
        exit()

    # Render the predicted video (Ahat0)
    ahat0_npy_path = os.path.join(run_dir, 'Ahat0.npy')
    ahat0_video_path = os.path.join(VIDEO_SAVE_DIR, 'predicted_video_{}.mp4'.format(RUN_TO_RENDER))
    render_video_from_npy(ahat0_npy_path, ahat0_video_path, VIDEO_FPS)

    # Render the original video for comparison
    original_npy_path = os.path.join(run_dir, 'original_video.npy')
    original_video_path = os.path.join(VIDEO_SAVE_DIR, 'original_video_{}.mp4'.format(RUN_TO_RENDER))
    render_video_from_npy(original_npy_path, original_video_path, VIDEO_FPS)

    print("\nRendering complete. Videos saved in: {}".format(VIDEO_SAVE_DIR))
