import numpy as np
import os
import h5py

#open h5 
SUMdata1 = h5py.File('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/model_data_keras2/tensorflow_weights/prednet_kitti_weights-Lall.hdf5', 'r')
# SUMdata2 = h5py.File('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer5_pool/seg4_Ahat.h5', 'r')
SUMdata1.keys()
# SUMdata2.keys()
# SUMdata=SUMdata['X_hat']



# data = h5py.File(filename,'r')
# data.keys() 
# data=data

# # SUMdata = SUMdata['model_weights']

## add additional npy to existing h5 file 
npy_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_3_32_64_96_192channel/'
hdf5_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_3_32_64_96_192channel/h5'
features = [ 'Ahat','A','E']
# num_of_features = 4 
file_names = ['seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7', 'seg8', 'seg9',
             'seg10', 'seg11', 'seg12', 'seg13', 'seg14', 'seg15', 'seg16', 'seg17', 
             'seg18', 'test1', 'test2', 'test3', 'test4', 'test5']
for file_name in file_names:
    for feature in features:
        hdf5_file_path = os.path.join(hdf5_dir, '{}_{}.h5'.format(file_name, feature))
        with h5py.File(hdf5_file_path, 'r') as data_check:
            # Retrieve and list all top-level keys/groups in the HDF5 file
            keys = list(data_check.keys())
            for key in keys:
                print('keys with shape: {}:{}'.format(key, data_check[key].shape))





haha