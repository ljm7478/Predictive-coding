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

from prednet_CNN_like import PredNet
# from haha2 import PredNet
from data_utils import SequenceGenerator
from step00_kitti_settings import *
import hickle as hkl

# layers4_output_modes = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3',  
#                         'E0', 'E1', 'E2','E3'] 

layers5_output_modes = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'Ahat4',
                        'E0', 'E1', 'E2','E3', 'E4'] 


batch_size = 4

# load titanic weight model weights 
weights_file = os.path.join('{}/CNN_like/Titanic_train_250epoch_layers5_32channel.hdf5'.format(Titanic_WEIGHTS_DIR)) 
json_file = os.path.join('{}/CNN_like/Titanic_train_250epoch_layers5_32channel.json'.format(Titanic_WEIGHTS_DIR)) 

# load test data
test_file = os.path.join('{}/X_12_Cam-CAN_movie.hkl'.format(CC_HICKLE_DUMP)) 
test_sources = os.path.join('{}/sources_12_Cam-CAN_movie.hkl'.format(CC_HICKLE_DUMP)) 
print(test_file, test_sources)

#nt 
nt = hkl.load(test_file)
nt = nt.shape[0]

for output_mode in layers5_output_modes:
    print('output_mode is '+ str(output_mode))

    save_dir= os.path.join('{}/modified_prednet/npy/layer5/'.format(CC_Results_DIR))
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
    train_model.load_weights(weights_file)

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

    test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
    X_test = test_generator.create_all()
    X_hat = test_model.predict(X_test, batch_size)
    if data_format == 'channels_first':
        X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))                                 
    
    # save prediction
    save_Xhat= os.path.join(save_dir,  str(layer_config['output_mode']) + '.npy')
    np.save(save_Xhat, X_hat)
    
    print('start next output_mode')

