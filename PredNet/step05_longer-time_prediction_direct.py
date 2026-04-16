#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Scenario 1: Direct Prediction - RTX 5090 VERSION
=================================================
Python 2.7 Compatible Version

Optimized for RTX 5090 (32GB) with batch_size=4
'''

from __future__ import print_function, division, absolute_import

import os

# ============================================================================
# GPU SELECTION - CHANGE THIS IF NEEDED!
# ============================================================================
# Try '1' first. If wrong GPU, change to '2'
GPU_ID = '1'  # <-- CHANGE TO '2' IF NEEDED

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
# ============================================================================

import numpy as np
import gc
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input

from prednet import PredNet
from data_utils import SequenceGenerator
from step00_kitti_settings import *
import hickle as hkl

# ============================================================================
# GPU CHECK
# ============================================================================
print('\n' + '='*50)
print('GPU CHECK - Using GPU {0}'.format(GPU_ID))
print('='*50)
if K.backend() == 'tensorflow':
    try:
        import tensorflow as tf
        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()
        for d in devices:
            if d.device_type == 'GPU':
                print('GPU: {0}'.format(d.name))
                print('Memory: {0:.1f} GB'.format(d.memory_limit / 1e9))
    except Exception as e:
        print('GPU check error: {0}'.format(e))
print('='*50 + '\n')

# ============================================================================
# CONFIGURATION
# ============================================================================

orig_weights_file = os.path.join('{0}/prednet/pt_kitti_ft_Lall_nt150/layer4/weights.hdf5'.format(Titanic_WEIGHTS_DIR))
orig_json_file = os.path.join('{0}/prednet/pt_kitti_ft_Lall_nt150/layer4/model.json'.format(Titanic_WEIGHTS_DIR))

GUMP_Results_DIR = './data/01_gump/longer-time_result'

runs = ['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8']

layers4_output_modes = [
    'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3',
    'E0', 'E1', 'E2', 'E3'
]

# ============================================================================
# DIRECT PREDICTION CONFIGS
# ============================================================================

WINDOW_SIZE = 10

SCIENTIFIC_CONFIGS = [
    {
        'skip_frames': 25,
        'window_size': WINDOW_SIZE,
        'name': 'direct_1s_pred',
        'horizon_sec': 1.0,
    },
    {
        'skip_frames': 50,
        'window_size': WINDOW_SIZE,
        'name': 'direct_2s_pred',
        'horizon_sec': 2.0,
    },
    {
        'skip_frames': 125,
        'window_size': WINDOW_SIZE,
        'name': 'direct_5s_pred',
        'horizon_sec': 5.0,
    },
    {
        'skip_frames': 250,
        'window_size': WINDOW_SIZE,
        'name': 'direct_10s_pred',
        'horizon_sec': 10.0,
    },
]

CONFIGS_TO_RUN = SCIENTIFIC_CONFIGS

# ============================================================================
# BATCH SIZE FOR RTX 5090 (32GB)
# ============================================================================
BATCH_SIZE = 4  # Safe for 32GB GPU


# ============================================================================
# OPTIMIZED DIRECT PREDICTION
# ============================================================================

def process_direct_optimized(X_full, skip_frames, window_size, test_model, 
                              data_format, batch_size):
    """
    Process all positions using batched prediction.
    Model is loaded ONCE, not per position.
    """
    total_frames = X_full.shape[0]
    
    # Get output shape (with proper transpose)
    sample_indices = [min(i * skip_frames, total_frames - 1) for i in range(window_size)]
    sample_window = np.expand_dims(X_full[sample_indices], axis=0)
    if data_format == 'channels_first':
        sample_window = np.transpose(sample_window, (0, 1, 4, 2, 3))
    sample_output = test_model.predict(sample_window, batch_size=1)
    
    # Transpose sample output to get correct shape for storage
    if data_format == 'channels_first':
        sample_output = np.transpose(sample_output, (0, 1, 3, 4, 2))
    
    # Now initialize with correct (transposed) shape
    single_frame_shape = sample_output[0, 0].shape
    output_shape = (1, total_frames) + single_frame_shape
    all_predictions = np.zeros(output_shape, dtype=np.float32)
    
    print('    Output shape will be: {0}'.format(output_shape))
    print('    Processing {0} positions (batch_size={1})...'.format(total_frames, batch_size))
    start_time = time.time()
    
    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        current_batch_size = batch_end - batch_start
        
        # Progress update every 500 positions
        if batch_start % 500 == 0 or batch_end == total_frames:
            elapsed = time.time() - start_time
            progress = batch_end / float(total_frames)
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            print('      Position {0}/{1} ({2:.1f}%) - ETA: {3:.1f}s'.format(
                batch_end, total_frames, progress * 100, eta))
        
        # Create batch of windows
        batch_windows = []
        for t in range(batch_start, batch_end):
            indices = [min(t + i * skip_frames, total_frames - 1) for i in range(window_size)]
            batch_windows.append(X_full[indices])
        
        batch_windows = np.array(batch_windows)
        
        if data_format == 'channels_first':
            batch_windows = np.transpose(batch_windows, (0, 1, 4, 2, 3))
        
        # Predict batch
        batch_pred = test_model.predict(batch_windows, batch_size=current_batch_size)
        
        if data_format == 'channels_first':
            batch_pred = np.transpose(batch_pred, (0, 1, 3, 4, 2))
        
        # Store first frame of each prediction
        all_predictions[0, batch_start:batch_end] = batch_pred[:, 0]
        
        del batch_windows, batch_pred
    
    total_time = time.time() - start_time
    print('    Done! Total time: {0:.1f}s'.format(total_time))
    
    return all_predictions


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('\n' + '='*70)
    print('DIRECT PREDICTION - RTX 5090 VERSION')
    print('GPU: {0}, Batch Size: {1}'.format(GPU_ID, BATCH_SIZE))
    print('='*70)
    
    total_start = time.time()
    
    runs_to_process = runs[0:8]
    
    print('\nRuns: {0}'.format(runs_to_process))
    print('Configs: {0}'.format([c['name'] for c in CONFIGS_TO_RUN]))
    
    # Load model architecture ONCE
    print('\nLoading model architecture...')
    f = open(orig_json_file, 'r')
    orig_json_string = f.read()
    f.close()
    
    for config in CONFIGS_TO_RUN:
        config_start = time.time()
        
        skip_frames = config['skip_frames']
        window_size = config['window_size']
        output_name = config['name']
        
        print('\n' + '#'*70)
        print('# {0}'.format(output_name.upper()))
        print('# skip_frames={0}, window_size={1}'.format(skip_frames, window_size))
        print('# Prediction horizon: {0}s'.format(config['horizon_sec']))
        print('#'*70)
        
        for run in runs_to_process:
            run_start = time.time()
            
            print('\n' + '='*60)
            print('Processing: {0} - {1}'.format(output_name, run))
            print('='*60)
            
            test_file = os.path.join(GUMP_HICKLE_DUMP, 'X_{0}.hkl'.format(run))
            
            if not os.path.exists(test_file):
                print('ERROR: File not found: {0}'.format(test_file))
                continue
            
            # Create save directory
            save_dir = os.path.join(
                GUMP_Results_DIR,
                '{0}/npy/pt_kitti_ft_Lall_nt150/layer4/{1}'.format(output_name, run)
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Load video
            print('  Loading video...')
            X_full = hkl.load(test_file)
            total_frames = X_full.shape[0]
            print('  Frames: {0}'.format(total_frames))
            
            # Save original once
            orig_path = os.path.join(save_dir, 'original_video.npy')
            if not os.path.exists(orig_path):
                np.save(orig_path, np.expand_dims(X_full, axis=0))
            
            # Process each output mode
            for output_mode in layers4_output_modes:
                print('\n  Output: {0}'.format(output_mode))
                
                save_path = os.path.join(save_dir, '{0}.npy'.format(output_mode))
                if os.path.exists(save_path):
                    print('    Already exists, skipping...')
                    continue
                
                # Clear session and build model ONCE
                K.clear_session()
                gc.collect()
                
                mode_start = time.time()
                
                # Load model
                orig_model = model_from_json(orig_json_string, custom_objects={'PredNet': PredNet})
                orig_model.load_weights(orig_weights_file)
                
                # Configure (NO extrap_start_time for direct)
                layer_config = orig_model.layers[1].get_config()
                layer_config['output_mode'] = output_mode
                
                data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
                
                # Build model
                test_prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)
                
                input_shape = list(orig_model.layers[0].batch_input_shape[1:])
                input_shape[0] = window_size
                
                inputs = Input(shape=tuple(input_shape))
                predictions = test_prednet(inputs)
                test_model = Model(inputs=inputs, outputs=predictions)
                
                # Process all positions with safe batch size
                X_hat = process_direct_optimized(
                    X_full, skip_frames, window_size, test_model, 
                    data_format, batch_size=BATCH_SIZE
                )
                
                # Save
                np.save(save_path, X_hat)
                print('    Saved: {0}'.format(save_path))
                print('    Shape: {0}'.format(X_hat.shape))
                print('    Time: {0:.1f}s'.format(time.time() - mode_start))
                
                del X_hat
            
            del X_full
            gc.collect()
            
            run_time = time.time() - run_start
            print('\n  Run {0} complete: {1:.1f}s ({2:.1f} min)'.format(
                run, run_time, run_time / 60))
        
        config_time = time.time() - config_start
        print('\n  Config {0} complete: {1:.1f} min ({2:.1f} hours)'.format(
            output_name, config_time / 60, config_time / 3600))
    
    total_time = time.time() - total_start
    print('\n' + '='*70)
    print('ALL COMPLETE!')
    print('Total time: {0:.1f} min ({1:.1f} hours)'.format(total_time / 60, total_time / 3600))
    print('='*70)


if __name__ == '__main__':
    main()