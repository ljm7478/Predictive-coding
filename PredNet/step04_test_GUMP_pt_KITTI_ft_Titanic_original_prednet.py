'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from numpy.lib.npyio import save
from six.moves import cPickle
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image as im
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
# from keras.models import model_from_json
# from jlee_training_for_debug import Model
from keras.layers import Input, Dense, Flatten
from keras import models
from keras.models import Sequential

from prednet import PredNet
from data_utils import SequenceGenerator
from step00_kitti_settings import *
import hickle as hkl

# layers4_output_modes = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3',  
#                         'E0', 'E1', 'E2','E3','C0', 'C1', 'C2', 'C3'] 
# layers5_output_modes = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'Ahat4', 
#                         'E0', 'E1', 'E2','E3', 'E4'] 
layers4_output_modes = ['O0', 'O1', 'O2', 'O3', 
                        'R0', 'R1', 'R2', 'R3']

runs = ['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8']
# runs = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7'] 

# load titanic weight model weights 
weights_file = os.path.join('{}/prednet/pt_kitti_ft_Lall_nt150/layer4/weights.hdf5'.format(Titanic_WEIGHTS_DIR)) 
json_file = os.path.join('{}/prednet/pt_kitti_ft_Lall_nt150/layer4/model.json'.format(Titanic_WEIGHTS_DIR)) 

# --- Main Evaluation Loop ---
print("Loading base model architecture from: {}".format(json_file))
with open(json_file, 'r') as f:
    json_string = f.read()
train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})

print("Loading fine-tuned weights from: {}".format(weights_file))
train_model.load_weights(weights_file)

GUMP_Results_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result'
for run in runs[0:1]:
    print("\nProcessing run: {}".format(run))
    # --- Load Test Data ---
    test_file = os.path.join(GUMP_HICKLE_DUMP, 'X_{}.hkl'.format(run))
    test_sources = os.path.join(GUMP_HICKLE_DUMP, 'sources_{}.hkl'.format(run))
    
    X_test_full = hkl.load(test_file)
    nt = X_test_full.shape[0]
    
    del X_test_full # Free up memory

    for output_mode in layers4_output_modes:
        print('output_mode is '+ str(output_mode))

        save_dir= os.path.join('{}/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer4/{}'.format(GUMP_Results_DIR, run))
        print(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Create testing model (to output predictions)
        layer_config = train_model.layers[1].get_config()
        layer_config['output_mode'] = output_mode
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
        
        input_shape = list(train_model.layers[0].batch_input_shape[1:])
        input_shape[0] = nt
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)

        # Create a generator that yields the entire video as a single sequence
        test_generator = SequenceGenerator(test_file, test_sources, nt, batch_size=1, sequence_start_mode='unique', N_seq=1, data_format=data_format)
        X_test = test_generator.create_all()
        X_hat = test_model.predict(X_test, batch_size=1)

        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))                                 
        
        # Save prediction
        save_Xhat= os.path.join(save_dir, str(output_mode) + '.npy')
        np.save(save_Xhat, X_hat)
        print("Saved to: {}".format(save_Xhat))

        # Save original image for first layer only
        if output_mode == 'Ahat0':
            save_Xtest = os.path.join(save_dir, 'original_video.npy')
            np.save(save_Xtest, X_test)
            
            print('Saved data for output_mode: {}'.format(layer_config["output_mode"]))
        
        print('start next output_mode')
print('\nEvaluation complete.')


