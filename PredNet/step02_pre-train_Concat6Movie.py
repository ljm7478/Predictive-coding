# -*- coding: utf-8 -*-
'''
Train PredNet on concatenated movie sequences (4 Layers, Lall Loss).
Uses updated hyperparameters suitable for the larger dataset.
'''

import os
import numpy as np
np.random.seed(123)
import pandas as pd

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator

# --- Configuration ---
# Path to the directory containing the split movie data
DATA_DIR = './data/06_concat6movie/hkl'
# Path to save the weights for this specific 4-layer model
WEIGHTS_DIR = './data/06_concat6movie/weights' # Specific directory for 4 layers

save_model = True
# Corrected filenames to reflect 4 layers
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_movie_4layers_Lall_nt10_epoch30_weights_Lall.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_movie_4layers_Lall_nt10_epoch30_model_Lall.json')
history_file = os.path.join(WEIGHTS_DIR, 'prednet_movie_4layers_Lall_nt10_epoch30_history.csv')

# Data files (using the '_new' split files)
train_file = os.path.join(DATA_DIR, 'X_train_new.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train_new.hkl')
val_file = os.path.join(DATA_DIR, 'X_val_new.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val_new.hkl')

# --- training parameters ---
nb_epoch = 150
batch_size = 8
samples_per_epoch = 5000 
N_seq_val = 1000         
nt = 10 

# # --- mid-success training parameters ---
# nb_epoch = 150
# batch_size = 8
# samples_per_epoch = 5000 
# N_seq_val = 1000         
# nt = 10   

# nb_epoch = 30
# batch_size = 8
# samples_per_epoch = 50000 
# N_seq_val = 5000         
# nt = 10 

## training fail parameters
# nb_epoch = 150
# batch_size = 4
# samples_per_epoch = 2000 
# N_seq_val = 500         
# nt = 50                 

# --- Model parameters for 4 Layers ---
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
# Standard 4-layer channel stack from KITTI pre-trained model
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)             # Length: num_layers - 1 = 3
Ahat_filt_sizes = (3, 3, 3, 3)       # Length: num_layers = 4
R_filt_sizes = (3, 3, 3, 3)          # Length: num_layers = 4
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1]) # Lall for 4 layers
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
# --- End Model parameters ---

time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True,
                  data_format=K.image_data_format()) # Using K backend format

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)
errors_by_time = Flatten()(errors_by_time)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)
model = Model(inputs=inputs, outputs=final_errors)

optimizer = 'adam' # Adam default lr is 0.001
model.compile(loss='mean_absolute_error', optimizer=optimizer)

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True, data_format=K.image_data_format())
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val, data_format=K.image_data_format())

# Original learning rate schedule
# lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001

# fixed epoch accordance with nb_epoch hyperparameter (nb_epoch=30)
lr_schedule = lambda epoch: 0.001 if epoch < 15 else 0.0001
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.makedirs(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, steps_per_epoch = samples_per_epoch // batch_size, epochs = nb_epoch, callbacks=callbacks,
                            validation_data=val_generator, validation_steps = N_seq_val // batch_size)

# --- Save Model and History ---
if save_model:
    # Save model architecture to JSON
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_file, index=False)


print("Pre-training complete for 4 layers (Lall). Weights saved to: {}".format(weights_file))
