'''

Train PredNet on Titanic sequences, starting from pre-trained KITTI weights.

'''

import os
import numpy as np
np.random.seed(123)
import pandas as pd

from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from prednet_CNN_like_layer5_nopool import PredNet
from data_utils import SequenceGenerator
from step00_kitti_settings import *

# --- File Paths ---
# Pre-trained model files
pretrained_json_file = './data/00_KITTI_pre-train/weights/CNN_like/prednet_kitti_5layers_nopool_model_Lall.json'   #channels_first
pretrained_weights_file = './data/00_KITTI_pre-train/weights/CNN_like/prednet_kitti_5layers_nopool_weights_Lall.hdf5'   #channels_first

# Where to save the fine-tuned model
save_model = True
output_dir = os.path.join(Titanic_WEIGHTS_DIR, 'CNN_like', 'pt_kitti_ft_Lall_nt150', 'layer5_nopool')
weights_file = os.path.join(output_dir, 'weights.hdf5')
json_file = os.path.join(output_dir, 'model.json')
history_file = os.path.join(output_dir, 'history.csv')

# Data files
train_file = os.path.join(Titanic_HICKLE_DUMP, 'X_train.hkl')
train_sources = os.path.join(Titanic_HICKLE_DUMP, 'sources_train.hkl')
val_file = os.path.join(Titanic_HICKLE_DUMP, 'X_val.hkl')
val_sources = os.path.join(Titanic_HICKLE_DUMP, 'sources_val.hkl')

# --- Training Hyperparameters ---
nb_epoch = 60
batch_size = 4
samples_per_epoch = 500
N_seq_val = 500
nt = 150


# --- Model Loading and Rebuilding ---
# Load pre-trained model architecture
with open(pretrained_json_file, 'r') as f:
    json_string = f.read()
model = model_from_json(json_string, custom_objects={'PredNet': PredNet})

# Load pre-trained weights
model.load_weights(pretrained_weights_file)

# Create new PredNet layer, inheriting config and weights from the pre-trained layer
layer_config = model.layers[1].get_config()
layer_config['output_mode'] = 'error'
layer_config['return_sequences'] = True

prednet = PredNet(weights=model.layers[1].get_weights(), **layer_config)

# Rebuild the model graph with the new sequence length (nt)
original_input_shape = model.layers[0].batch_input_shape[1:] # (10, 3, 128, 160)
input_shape = (nt,) + original_input_shape[1:] # (200, 3, 128, 160)
inputs = Input(shape=input_shape)

# Define loss calculation layers
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1, 0.1])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1. / (nt - 1) * np.ones((nt, 1))
time_loss_weights[0] = 0

errors = prednet(inputs)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)
errors_by_time = Flatten()(errors_by_time)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)

finetune_model = Model(inputs=inputs, outputs=final_errors)

# Compile model for fine-tuning
finetune_model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0001))

# --- Data Generators and Callbacks ---
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True, data_format='channels_last')
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val, data_format='channels_last')

callbacks = []
if save_model:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

# --- Start Training ---
history = finetune_model.fit_generator(train_generator, steps_per_epoch=samples_per_epoch // batch_size, epochs=nb_epoch, callbacks=callbacks,
                                     validation_data=val_generator, validation_steps=N_seq_val // batch_size)

# --- Save Model and History ---
if save_model:
    # Save model architecture to JSON
    json_string = finetune_model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

    # Save training history to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_file, index=False)

print("Fine-tuning complete! Best weights saved to {}".format(weights_file))
print("Training history saved to {}".format(history_file))