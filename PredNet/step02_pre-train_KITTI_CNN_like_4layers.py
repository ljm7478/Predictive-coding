# -*- coding: utf-8 -*-
'''
Train PredNet on KITTI sequences (5 Layers, L0 Loss).
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

from prednet_CNN_like import PredNet
from data_utils import SequenceGenerator

# --- Configuration ---
#### docker path
DATA_DIR = './data/00_KITTI_pre-train/hkl'     
WEIGHTS_DIR = './data/00_KITTI_pre-train/weights' 
MODEL_SUB_DIR = 'CNN_like' 
SAVE_DIR = os.path.join(WEIGHTS_DIR, MODEL_SUB_DIR) 

save_model = True
weights_file = os.path.join(SAVE_DIR, 'prednet_kitti_4layers_weights_Lall.hdf5') # Unique filename
json_file = os.path.join(SAVE_DIR, 'prednet_kitti_4layers_model_Lall.json') # Unique filename
history_file = os.path.join(SAVE_DIR, 'prednet_kitti_4layers_model_Lall_history.csv')

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters (same as original)
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100
nt = 10

# --- Model parameters for 4 Layers ---
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
# Added one more layer with 256 channels (adjust if needed)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)          # nb_layers - 1
Ahat_filt_sizes = (3, 3, 3, 3)     
R_filt_sizes = (3, 3, 3, 3)        
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1]) # Lall
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
# --- End Model parameters ---

time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True,
                  data_format=K.image_data_format())

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)
errors_by_time = Flatten()(errors_by_time)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)
model = Model(inputs=inputs, outputs=final_errors)

optimizer = 'adam'
model.compile(loss='mean_absolute_error', optimizer=optimizer)

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True, data_format=K.image_data_format())
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val, data_format=K.image_data_format())

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
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

print("Training complete for 4 layers (CNN_like). Weights saved to: {}".format(weights_file))
