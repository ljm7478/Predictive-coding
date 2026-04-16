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

# from 02_prednet_CNN_like import PredNet  
from prednet_CNN_like_alexnet_structure_follow import PredNet
# from haha2 import PredNet
from data_utils import SequenceGenerator
from step00_kitti_settings import *
import hickle as hkl

# dir path 
AlexNet_weight = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/weights'
AlexNet_base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet'

runs = ['seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7', 'seg8', 'seg9',
        'seg10', 'seg11', 'seg12', 'seg13', 'seg14', 'seg15', 'seg16', 'seg17', 
        'seg18', 'test1', 'test2', 'test3', 'test4', 'test5']



# layers5_output_modes = ['A1', 'A2', 'A3', 'A4' , 'Ahat1', 'Ahat2', 'Ahat3', 'Ahat4',  'E1', 'E2', 'E3','E4'] #, 'R1', 'R2', 'R3', 'R4', 'C1', 'C2', 'C3', 'C4'] 
layers5_output_modes = ['A0', 'E0'] 

batch_size = 4

for run in runs: 
    print(str(run))

    # # load titanic weight model weights 
    # #- 3, 96, 256, 384, 384, 256 channel
    weights_file = os.path.join('{}/Titanic_train_250epoch_layers5_96channel_alexnet_follow.hdf5'.format(AlexNet_weight)) 
    json_file = os.path.join('{}/Titanic_train_250epoch_layers5_96channel_alexnet_follow.json'.format(AlexNet_weight)) 
    
    # #- 3, 32, 64, 96, 192 channel
    # weights_file = os.path.join('{}/Titanic_train_250epoch_layers5_96channel_02_prednet_CNN_like.hdf5'.format(AlexNet_weight)) 
    # json_file = os.path.join('{}/Titanic_train_250epoch_layers5_96channel_02_prednet_CNN_like.json'.format(AlexNet_weight)) 
    
    # load test data
    # 3, 96, 256, 384, 384, 256 channel
    test_file = os.path.join('{}/hkl/img_227_227/X_{}.hkl'.format(AlexNet_base_dir, str(run))) 
    test_sources = os.path.join('{}/hkl/img_227_227/sources_{}.hkl'.format(AlexNet_base_dir, str(run))) 
    print(test_file, test_sources)
    
    # 3, 32, 64, 96, 192 channel channel
    # test_file = os.path.join('{}/hkl/img_128_160/X_{}.hkl'.format(AlexNet_base_dir, str(run))) 
    # test_sources = os.path.join('{}/hkl/img_128_160/sources_{}.hkl'.format(AlexNet_base_dir, str(run))) 
    # print(test_file, test_sources)
    
    #nt 
    nt = hkl.load(test_file)
    nt = nt.shape[0]

    for output_mode in layers5_output_modes:
        print('output_mode is '+ str(output_mode))

        save_dir= os.path.join('{}/result/modified_prednet/layers5_96_256_384_384_256channel/{}'.format(AlexNet_base_dir, str(run)))
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
        # if X_test[0].shape != X_test[2].shape:
        #     print("Arrays have different shapes.")
        # else:
        #     # Find where arrays differ
        #     diff_indices = np.where(X_test[0] != X_test[2])
            
        #     # Print the indices where arrays have different values
        #     if len(diff_indices[0]) > 0:
        #         print("Arrays differ at the following indices:")
        #         for i in range(len(diff_indices[0])):
        #             print("Index", diff_indices[0][i], "(", X_test[0][diff_indices[0][i]], "vs", X_test[2][diff_indices[0][i]], ")")
        #     else:
        #         print("Arrays are identical.")
            
        X_hat = test_model.predict(X_test, batch_size)
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))                                    

        save_Xhat= os.path.join(save_dir + '/',  str(layer_config['output_mode']) + '.npy')
        np.save(save_Xhat,X_hat)
        
        print('X_hat shape is ', str(layer_config['output_mode']) + '_' +  str(X_hat.shape))
    

        # Plot some predictions
        # n_plot=10
        # nt=10
        # aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        # plt.figure(figsize = (nt, 2*aspect_ratio))
        # gs = gridspec.GridSpec(2, nt) 
        # gs.update(wspace=0., hspace=0.)
        # # plot_save_dir = os.path.join(save_dir, 'prediction_plots/')
        # # if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)

        # # plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
        # # i=0
        # # X_test = np.swapaxes(X_test, 0, 1)
        # # X_hat = np.swapaxes(X_hat, 0, 1)
        # # # for i in plot_idx:
        # # for t in range(nt):
        # #     X_test1 = np.squeeze(X_test, -1)
        # #     # X_test = np.swapaxes(X_test, 0, 1)
        # #     plt.subplot(gs[t])
        # #     plt.imshow(X_test1[i,t], interpolation='none')
        # #     plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        # #     if t==0: plt.ylabel('Actual', fontsize=10)

        # #     # X_hat = X_hat[:,:,:,:,:3]
        # #     # X_hat = np.swapaxes(X_hat, 0, 1)
        # #     plt.subplot(gs[t+nt])
        # #     X_hat1 = X_hat[:,:,:,:,:1]
        # #     X_hat1 = np.squeeze(X_hat1, -1)
        # #     plt.imshow(X_hat1[i,t], interpolation='none')
        # #     plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        # #     if t==0: plt.ylabel('Error', fontsize=10)
            
        # #     # plt.subplot(gs[2, t])
        # #     # X_hat2 = X_hat[:,:,:,:,1:2]
        # #     # X_hat2 = np.squeeze(X_hat2, -1)
        # #     # plt.imshow(X_hat2[i,t], interpolation='none', cmap='gray')
        # #     # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        # #     # if t==0: plt.ylabel('Error', fontsize=10)

        # # # plt.savefig(save_dir + str(output_mode)+ '_plot_11' + str(i) + '.png')
        # plt.clf()
        
        # plot_idx = [0]
        # for i in plot_idx:
        #     for t in range(nt):
        #         plt.subplot(gs[t])
        #         plt.imshow(X_test[i,t], interpolation='none')
        #         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        #         if t==0: plt.ylabel('Actual', fontsize=10)

        #         plt.subplot(gs[t + nt])
        #         plt.imshow(X_hat[i,t], interpolation='none')
        #         plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        #         if t==0: plt.ylabel(str(output_mode), fontsize=10)

        #     # plt.savefig(plot_save_dir + str(output_mode)+ '_plot_' + str(i) + '.png')
        #     plt.clf()
        
        print('start next output_mode')

