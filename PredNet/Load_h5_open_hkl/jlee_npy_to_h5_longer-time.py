import numpy as np
import h5py
import os
                
####predimg  - GUMP test 

model_type = ['prednet_original', 'CNN_like']
layer_type = ['layer4', 'layer5_nopool']
run_names = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7'] 

features = ['Ahat', 'E']
# features = ['C']

for model in model_type[0:1]: 
    for layer in layer_type[0:1]: 
        if layer == 'layer4':
            num_of_features = 4 
        else:
            num_of_features = 5
        hdf5_dir=os.path.join('./data/01_gump/longer-time_result/{}/h5/pt_kitti_ft_Lall_nt150/indirect_1s_extrap/{}'.format(model, layer))
        if not os.path.exists(hdf5_dir):
            os.makedirs(hdf5_dir)
        for run in run_names[5:1]:
            npy_dir = os.path.join('./data/01_gump/longer-time_result/indirect_1s_extrap/npy/pt_kitti_ft_Lall_nt150/{}/run{}').format( layer, int(run[-1])+1)
            print(npy_dir)
            for feature in features[1:2]:
                hdf5_file_path = os.path.join(hdf5_dir, '{}_{}.h5'.format(run, feature))
                with h5py.File(hdf5_file_path, 'w') as hf:
                    for r in range(num_of_features):
                        feature_label = '{}{}'.format(feature, r)  # Label e.g., Ahat1
                        print('feature_label is {}'.format(feature_label))
                        
                        npy_file_path = os.path.join(npy_dir, '{}.npy'.format(feature_label))
                        print('npy_file_path is {}'.format(npy_file_path))
                        
                        if os.path.exists(npy_file_path):
                            # Load the numpy file
                            data = np.load(npy_file_path)
                            print('npy data shape: {}'.format(data.shape))
                            print('any negative in array: {}'.format(np.any(data<0)))
                            
                            # Create a dataset for each feature in the corresponding group
                            hf.create_dataset(feature_label, data=data) #, compression='gzip', compression_opts=9)
                            print('Saved {} in {}'.format(feature_label, hdf5_file_path))
                with h5py.File(hdf5_file_path, 'r') as data_check:
                    # Retrieve and list all top-level keys/groups in the HDF5 file
                    for i in range(num_of_features):
                        keys = list(data_check.keys())
                        print('keys with shape: {}:{}'.format(keys[i], data_check[keys[i]].shape))            

