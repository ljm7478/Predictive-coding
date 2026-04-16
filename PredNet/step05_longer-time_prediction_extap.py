#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Scenario 2: Indirect Prediction - OPTIMIZED VERSION
====================================================
Python 2.7 Compatible Version

OPTIMIZATION: Avoids re-initializing GPU for every chunk.
- Load model ONCE per output_mode
- Process ALL chunks with same model
- Only clear session between output_modes

This reduces processing time from ~3 hours/run to ~15 min/run!
'''

from __future__ import print_function, division, absolute_import

import os
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
print('GPU CHECK')
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

# layers4_output_modes = [
#     'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3',
#     'E0', 'E1', 'E2', 'E3'
# ]
layers4_output_modes = ['E1', 'E2', 'E3']

# ============================================================================
# SCIENTIFIC CONFIGURATIONS
# ============================================================================

CONTEXT_FRAMES = 25

SCIENTIFIC_CONFIGS = [
    {
        'chunk_size': 50,
        'extrap_start_time': CONTEXT_FRAMES,
        'name': 'indirect_1s_extrap',
        'extrap_frames': 25,
        'extrap_sec': 1.0,
    },
    # {
    #     'chunk_size': 75,
    #     'extrap_start_time': CONTEXT_FRAMES,
    #     'name': 'indirect_2s_extrap',
    #     'extrap_frames': 50,
    #     'extrap_sec': 2.0,
    # },
    # {
    #     'chunk_size': 150,
    #     'extrap_start_time': CONTEXT_FRAMES,
    #     'name': 'indirect_5s_extrap',
    #     'extrap_frames': 125,
    #     'extrap_sec': 5.0,
    # },
    # {
    #     'chunk_size': 275,
    #     'extrap_start_time': CONTEXT_FRAMES,
    #     'name': 'indirect_10s_extrap',
    #     'extrap_frames': 250,
    #     'extrap_sec': 10.0,
    # },
]

CONFIGS_TO_RUN = SCIENTIFIC_CONFIGS


# ============================================================================
# OPTIMIZED PROCESSING - NO GPU REINIT PER CHUNK
# ============================================================================

def process_all_chunks_single_session(X_full, chunk_size, extrap_start_time, 
                                       test_model, data_format):
    """
    Process ALL chunks using a SINGLE model (no GPU reinit).
    
    The model is built for chunk_size. For the last chunk (possibly smaller),
    we pad it to chunk_size.
    """
    total_frames = X_full.shape[0]
    num_chunks = int(np.ceil(total_frames / float(chunk_size)))
    
    all_predictions = []
    
    print('    Processing {0} frames in {1} chunks...'.format(total_frames, num_chunks))
    start_time = time.time()
    
    for chunk_idx in range(num_chunks):
        start_frame = chunk_idx * chunk_size
        end_frame = min(start_frame + chunk_size, total_frames)
        actual_size = end_frame - start_frame
        
        # Extract chunk
        chunk = X_full[start_frame:end_frame]
        
        # Pad if last chunk is smaller
        if actual_size < chunk_size:
            pad_size = chunk_size - actual_size
            padding = np.repeat(chunk[-1:], pad_size, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)
        
        # Add batch dimension
        chunk_input = np.expand_dims(chunk, axis=0)
        
        if data_format == 'channels_first':
            chunk_input = np.transpose(chunk_input, (0, 1, 4, 2, 3))
        
        # Predict (fast - no model reload!)
        chunk_pred = test_model.predict(chunk_input, batch_size=1)
        
        if data_format == 'channels_first':
            chunk_pred = np.transpose(chunk_pred, (0, 1, 3, 4, 2))
        
        # Remove padding from output
        chunk_pred = chunk_pred[:, :actual_size]
        
        all_predictions.append(chunk_pred)
        
        # Progress update every 50 chunks
        if (chunk_idx + 1) % 50 == 0 or chunk_idx == num_chunks - 1:
            elapsed = time.time() - start_time
            eta = elapsed / (chunk_idx + 1) * (num_chunks - chunk_idx - 1)
            print('      Chunk {0}/{1} - Elapsed: {2:.1f}s, ETA: {3:.1f}s'.format(
                chunk_idx + 1, num_chunks, elapsed, eta))
    
    # Concatenate
    result = np.concatenate(all_predictions, axis=1)
    total_time = time.time() - start_time
    print('    Done! Total time: {0:.1f}s ({1:.2f}s per chunk)'.format(
        total_time, total_time / num_chunks))
    
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('\n' + '='*70)
    print('INDIRECT PREDICTION - OPTIMIZED (NO GPU REINIT PER CHUNK)')
    print('='*70)
    
    total_start = time.time()
    
    runs_to_process = runs[0:1]
    
    print('\nRuns: {0}'.format(runs_to_process))
    print('Configs: {0}'.format([c['name'] for c in CONFIGS_TO_RUN]))
    
    # Load model architecture ONCE
    print('\nLoading model architecture...')
    f = open(orig_json_file, 'r')
    orig_json_string = f.read()
    f.close()
    
    for config in CONFIGS_TO_RUN:
        config_start = time.time()
        
        chunk_size = config['chunk_size']
        extrap_start_time = config['extrap_start_time']
        output_name = config['name']
        
        print('\n' + '#'*70)
        print('# {0}'.format(output_name.upper()))
        print('# chunk_size={0}, extrap_start={1}'.format(chunk_size, extrap_start_time))
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
            
            # Load video ONCE for all output modes
            print('  Loading video...')
            X_full = hkl.load(test_file)
            total_frames = X_full.shape[0]
            num_chunks = int(np.ceil(total_frames / float(chunk_size)))
            print('  Frames: {0}, Chunks: {1}'.format(total_frames, num_chunks))
            
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
                
                # Clear session and build model ONCE for this output_mode
                K.clear_session()
                gc.collect()
                
                mode_start = time.time()
                
                # Load model
                orig_model = model_from_json(orig_json_string, custom_objects={'PredNet': PredNet})
                orig_model.load_weights(orig_weights_file)
                
                # Configure
                layer_config = orig_model.layers[1].get_config()
                layer_config['output_mode'] = output_mode
                layer_config['extrap_start_time'] = extrap_start_time
                
                data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
                
                # Build model for chunk_size
                test_prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)
                
                input_shape = list(orig_model.layers[0].batch_input_shape[1:])
                input_shape[0] = chunk_size  # Fixed size
                
                inputs = Input(shape=tuple(input_shape))
                predictions = test_prednet(inputs)
                test_model = Model(inputs=inputs, outputs=predictions)
                
                # Process ALL chunks with this model (no reinit!)
                X_hat = process_all_chunks_single_session(
                    X_full, chunk_size, extrap_start_time, test_model, data_format
                )
                
                # Save
                np.save(save_path, X_hat)
                print('    Saved: {0}'.format(save_path))
                print('    Shape: {0}'.format(X_hat.shape))
                print('    Time: {0:.1f}s'.format(time.time() - mode_start))
                
                del X_hat
            
            # Clear video from memory
            del X_full
            gc.collect()
            
            run_time = time.time() - run_start
            print('\n  Run {0} complete: {1:.1f}s ({2:.1f} min)'.format(
                run, run_time, run_time / 60))
        
        config_time = time.time() - config_start
        print('\n  Config {0} complete: {1:.1f}s ({2:.1f} min)'.format(
            output_name, config_time, config_time / 60))
    
    total_time = time.time() - total_start
    print('\n' + '='*70)
    print('ALL COMPLETE!')
    print('Total time: {0:.1f}s ({1:.1f} hours)'.format(total_time, total_time / 3600))
    print('='*70)


if __name__ == '__main__':
    main()