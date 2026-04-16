import numpy as np
import h5py
import os
# from kitti_settings import *


### AlexNet - Neural encoding decoding method - save activations for all layers in one .h5 file per seg

# file_names = ['seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7', 'seg8', 'seg9',
#              'seg10', 'seg11', 'seg12', 'seg13', 'seg14', 'seg15', 'seg16', 'seg17', 
#              'seg18', 'test1', 'test2', 'test3', 'test4', 'test5']

# features = ['A', 'Ahat', 'E']
# num_of_features = 4 

# npy_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/'
# hdf5_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5'


# Process each segment
# for file_name in file_names:
#     for feature in features[1:3]:
#         hdf5_file_path = os.path.join(hdf5_dir, '{}_{}.h5'.format(file_name, feature))
#         # if not os.path.exists(hdf5_dir):
#         #     os.makedirs(hdf5_dir)
            
#         with h5py.File(hdf5_file_path, 'w') as hf:
#             for r in range(num_of_features):
#                 feature_label = '{}{}'.format(feature, r+1)  # Label e.g., Ahat1
#                 print('feature_label is {}'.format(feature_label))
                
#                 npy_file_path = os.path.join(npy_dir, file_name, '{}.npy'.format(feature_label))
#                 print('npy_file_path is {}'.format(npy_file_path))
                
#                 if os.path.exists(npy_file_path):
#                     # Load the numpy file
#                     data = np.load(npy_file_path)
#                     print('npy data shape: {}'.format(data.shape))
#                     print('any negative in array: {}'.format(np.any(data<0)))
#                     # Create a dataset for each feature in the corresponding group
#                     hf.create_dataset(feature_label, data=data) #, compression='gzip', compression_opts=9)
#                     print('Saved {} in {}'.format(feature_label, hdf5_file_path))

#         with h5py.File(hdf5_file_path, 'r') as data_check:
#             # Retrieve and list all top-level keys/groups in the HDF5 file
#             for i in range(num_of_features):
#                 keys = list(data_check.keys())
#                 print('keys with shape: {}:{}'.format(keys[i], data_check[keys[i]].shape))
                

###### add additional npy to existing h5 file 
# npy_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/'
# hdf5_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5'
# features = [ 'Ahat','A','E']
# # num_of_features = 4 

# for file_name in file_names:
#     for feature in features:
#         hdf5_file_path = os.path.join(hdf5_dir, '{}_{}.h5'.format(file_name, feature))
        
#         new_feature_label = '{}0'.format(feature)
#         print(new_feature_label)
        
#         label_name = '{}.npy'.format(new_feature_label)
#         new_npy_file_path = os.path.join(npy_dir, file_name, label_name)
        
#         if os.path.exists(new_npy_file_path):
#             with h5py.File(hdf5_file_path, 'a') as hf:
#             # Check if the dataset exists, and delete it if it does
#             if feature in hf:
#                 del hf[label_name]
#                 print('Deleted dataset {} from {}'.format(label_name, hdf5_file_path))
                
#             with h5py.File(hdf5_file_path, 'a') as hf:  # Open the HDF5 file in append mode
#                 # Load the new numpy file
#                 new_data = np.load(new_npy_file_path)
#                 print('new npy data shape: {}'.format(new_data.shape))
#                 print('any negative in array: {}'.format(np.any(new_data < 0)))
                
#                 if new_feature_label in hf:
#                     del hf[new_feature_label]  # Delete the existing dataset if it exists
                
#                 hf.create_dataset(new_feature_label, data=new_data)  # Create the new dataset
#                 print('Saved {} in {}'.format(new_feature_label, hdf5_file_path))
        
#         with h5py.File(hdf5_file_path, 'r') as data_check:
#             # Retrieve and list all top-level keys/groups in the HDF5 file
#             keys = list(data_check.keys())
#             for key in keys:
#                 print('keys with shape: {}:{}'.format(key, data_check[key].shape))
                
####predimg  - GUMP test 

model_type = ['prednet_original', 'CNN_like']
layer_type = ['layer4', 'layer7']
file_names = ['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8']
features = ['Ahat', 'E']

for model in model_type[0:1]: 
    for layer in layer_type: 
        for file_name in file_names:
            for feature in features:
                for r in range(0,4):
                    feature_label = '{}{}'.format(feature, r)  # Label e.g., Ahat1
                    print('feature_label is {}'.format(feature_label))
                    
                    npy_dir = os.path.join('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/{}/npy/{}/{}').format(model, layer, file_name)
                    print(npy_dir)
                    npy_file_path = os.path.join(npy_dir, '{}.npy'.format(feature_label))
                    print('npy_file_path is {}'.format(npy_file_path))
                    
                    hdf5_file_path = os.path.join('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/{}/h5/{}/{}').format(model, layer, file_name)
                    if not os.path.exists(hdf5_file_path):
                        os.makedirs(hdf5_file_path)
                    
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




# ###predimg  - CC test 
# features = ['Ahat', 'E']
# num_of_features = 5 

# npy_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_Cam-CAN/result/modified_prednet/npy/layer5/'
# hdf5_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_Cam-CAN/result/modified_prednet/h5/layer5'


# # Process each segment
# for feature in features:
#     hdf5_file_path = os.path.join(hdf5_dir, '{}.h5'.format(feature))
#     # if not os.path.exists(hdf5_dir):
#     #     os.makedirs(hdf5_dir)
        
#     with h5py.File(hdf5_file_path, 'w') as hf:
#         for r in range(num_of_features):
#             feature_label = '{}{}'.format(feature, r)  # Label e.g., Ahat1
#             print('feature_label is {}'.format(feature_label))
            
#             npy_file_path = os.path.join(npy_dir, '{}.npy'.format(feature_label))
#             print('npy_file_path is {}'.format(npy_file_path))
            
#             if os.path.exists(npy_file_path):
#                 # Load the numpy file
#                 data = np.load(npy_file_path)
#                 print('npy data shape: {}'.format(data.shape))
#                 print('any negative in array: {}'.format(np.any(data<0)))
#                 # Create a dataset for each feature in the corresponding group
#                 hf.create_dataset(feature_label, data=data) #, compression='gzip', compression_opts=9)
#                 print('Saved {} in {}'.format(feature_label, hdf5_file_path))

#     with h5py.File(hdf5_file_path, 'r') as data_check:
#         # Retrieve and list all top-level keys/groups in the HDF5 file
#         for i in range(num_of_features):
#             keys = list(data_check.keys())
#             print('keys with shape: {}:{}'.format(keys[i], data_check[keys[i]].shape))
            
###predimg  - jmlee test 

# stimulus_type_30fps = ['circle', 'garbor_frequency', 'garbor_orientation',  'garbor_sc']
# session_type_30fps = 'unpredictable_30fps'

# stimulus_type_20fps = ['movie']
# session_type_20fps = 'unpredictable_20fps'

# features = ['Ahat', 'E']
# num_of_features = 7

# base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp'

# # Process each segment
# for stimulus in stimulus_type_20fps:
#     hdf5_dir=os.path.join(base_dir,  stimulus, session_type_20fps, 'result/CNN_like/h5/layer7/')
#     if not os.path.exists(hdf5_dir):
#             os.makedirs(hdf5_dir)
#     for feature in features:   
#         hdf5_file_path = os.path.join(hdf5_dir, '{}.h5'.format(feature))
#         with h5py.File(hdf5_file_path, 'w') as hf:
#             for r in range(num_of_features):
#                 feature_label = '{}{}'.format(feature, r)  # Label e.g., Ahat1
#                 print('feature_label is {}'.format(feature_label))
                
#                 npy_dir=os.path.join(base_dir,  stimulus, session_type_20fps, 'result/CNN_like/layer7/')
#                 npy_file_path = os.path.join(npy_dir, '{}.npy'.format(feature_label))
#                 print('npy_file_path is {}'.format(npy_file_path))
                
#                 if os.path.exists(npy_file_path):
#                     # Load the numpy file
#                     data = np.load(npy_file_path)
#                     print('npy data shape: {}'.format(data.shape))
#                     print('any negative in array: {}'.format(np.any(data<0)))
#                     # Create a dataset for each feature in the corresponding group
#                     hf.create_dataset(feature_label, data=data) #, compression='gzip', compression_opts=9)
#                     print('Saved {} in {}'.format(feature_label, hdf5_file_path))

#         with h5py.File(hdf5_file_path, 'r') as data_check:
#             # Retrieve and list all top-level keys/groups in the HDF5 file
#             for i in range(num_of_features):
#                 keys = list(data_check.keys())
#                 print('keys with shape: {}:{}'.format(keys[i], data_check[keys[i]].shape))            

#####predimg  - HCP test 

# file_names = ['7T_MOVIE1_CC1_v2', '7T_MOVIE2_HO1_v2', '7T_MOVIE3_CC2_v2', '7T_MOVIE4_HO2_v2']

# features = ['Ahat', 'E']

# for file_name in file_names:
#     for feature in features:
#         for r in range(1, 4):
#             save_folder_name = file_name[3:9]
#             print(save_folder_name)

#             npydir = os.path.join('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers_48channel/npy/'+ save_folder_name + '/')
            
#             savedir = os.path.join('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers_48channel/h5/'+ save_folder_name + '/')
            
#             if not os.path.exists(savedir):
#                 os.makedirs(savedir)

#             fileEX = save_folder_name + '_' + feature + str(r) + ".npy"
#             print(fileEX)

#             predimg = np.load(os.path.join(npydir, fileEX))
#             print('predimg shape: ' + str(predimg.shape))

#             filename = os.path.join(savedir, save_folder_name + '_' + feature + str(r) + '.h5')
#             print('filename: ' + filename)

#             # Convert .npy to .h5
#             hf = h5py.File(filename, 'w')
#             hf.create_dataset('predimg', data=predimg)
#             hf.close()
#             print('finish')

#             # Check if predimg data is well converted into .h5 format
#             data = h5py.File(filename, 'r')
#             print(data.keys())
#             data.close()


        



# ##convert .npy to .h5
# for seg in range(1,19):
#     print('seg'+ str(seg))
#     segment ='seg'+ str(seg)

#     predimg = np.load(os.path.join(AlexEncoding_result_DIR,'npy', 'total_output', segment + '_pre_video.npy'))
#     print('filepath: ' + os.path.join(AlexEncoding_result_DIR,'npy',  segment + '_pre_video.npy'))
#     print('predimg shape: ' + str(predimg.shape))
    
#     preimg_reshape = np.transpose(predimg, (0,1,4,3,2))    
#     print('preimg_reshape: ' + str(preimg_reshape.shape))
   
#     filename = AlexEncoding_h5_DIR + segment +'.h5'
#     print('filename: ' + filename)

#     ##convert .npy to .h5
#     hf = h5py.File(filename, 'w')
#     hf.create_dataset('predimg', data=preimg_reshape)
#     hf.close()
#     print('finish')

#     # ##check if predimg data is well converted into .h5 format
#     # data = h5py.File(filename,'r')
#     # data.keys()
#     # data=data


# #predimg  - NNDB
# for r in range(0,4):
#     npydir = os.path.join('/local_raid1/03_user/jungmin/01_project/01_PredNet/jmPNET/data/08_NNDB/result' +'/' +'npy' + '/' + 'E') #config[:-1])
#     savedir = os.path.join('/local_raid1/03_user/jungmin/01_project/01_PredNet/jmPNET/data/08_NNDB/result' + '/'+ 'h5' + '/' + 'E') # config[:-1])
#     print(npydir)
#     fileEX = 'E'+ str(r) + "_pre_video.npy"  # 'seg' + str(r) + '_'+  #fileEX = str(config_name[1:])+"_"+"train23456_2.npy"
#     print(fileEX)
    
#     predimg = np.load(os.path.join(npydir,fileEX))
#     print('predimg shape: ' + str(predimg.shape))
    
#     filename = savedir + '/' + 'E' + str(r) + '.h5'
#     print('filename: ' + filename)

#     ##convert .npy to .h5
#     hf = h5py.File(filename, 'w')
#     hf.create_dataset('predimg', data=predimg)
#     hf.close()
#     print('finish')

#     # ##check if predimg data is well converted into .h5 format
#     # data = h5py.File(filename,'r')
#     # data.keys()
#     # data=data

# #predimg - use ONLY IF order of dimension changed 
# for seg in range(1,19):
#     for r in range(0,4):
#         npydir = os.path.join('/local_raid1/03_user/jungmin/01_project/01_PredNet/jmPNET/data/04_AlexNet_WenH_Encoding/230414_FG_trained_3000epoch_seg_h5/npy/per_seg/Ahat') #config[:-1])
#         savedir = os.path.join('/local_raid1/03_user/jungmin/01_project/01_PredNet/jmPNET/data/04_AlexNet_WenH_Encoding/230414_FG_trained_3000epoch_seg_h5/h5/Ahat') # config[:-1])
#         print(npydir)
        
#         fileEX = 'seg{}_Ahat{}_pre_video.npy'.format(seg, r)
#         print(fileEX)
        
#         predimg = np.load(os.path.join(npydir,fileEX))
#         print('predimg shape: ' , predimg.shape)
        
#         if seg > 1 :
#             predimg_reshape = np.transpose(predimg,(1,0,2,3,4))
#             print('preimg_reshape: ', predimg_reshape.shape)

#             filename = os.path.join(savedir, 'seg{}_Ahat{}.h5'.format(seg, r))
#             print('filename: ' + filename)

#             ##convert .npy to .h5
#             hf = h5py.File(filename, 'w')
#             hf.create_dataset('predimg', data=predimg_reshape)
#             hf.close()
#             print('finish') 
#         else :
#             filename = os.path.join(savedir, 'seg{}_Ahat{}.h5'.format(seg, r))
#             print('filename: ' + filename)

#             ##convert .npy to .h5
#             hf = h5py.File(filename, 'w')
#             hf.create_dataset('predimg', data=predimg)
#             hf.close()
#             print('finish') 

    # ##check if predimg data is well converted into .h5 format
    # data = h5py.File(filename,'r')
    # data.keys()
    # data=data