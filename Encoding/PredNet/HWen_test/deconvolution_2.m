clc; clear;

%% Load PredNet weight of 1st layer
hdf5_file = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/weights/Titanic_train_250epoch_layers5_96channel_alexnet_follow.hdf5';

% Paths to the specific datasets within the HDF5 file
bias_path = '/model_weights/pred_net_1/pred_net_1/layer_ahat_1/bias:0';
kernel_path = '/model_weights/pred_net_1/pred_net_1/layer_ahat_1/kernel:0';

% Extract the bias and kernel
bias = h5read(hdf5_file, bias_path);
kernel = h5read(hdf5_file, kernel_path);

% Display the sizes to ensure correct extraction
disp(size(bias));   % Should be [96, 1]
disp(size(kernel)); % Should be [96, 96, 5, 5]

% Optionally, save the extracted weights and biases into a .mat file
savedir = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/deconvolution/';
save(fullfile(savedir, 'ahat1_weights.mat'), 'bias', 'kernel');

%%%%%%%%%%%%%%% Load test fMRI responses
subject_id = 'subject1';
dataroot_fmri = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];
load([dataroot_fmri, 'testing_fmri.mat'], 'fmritest');  % Load preprocessed fMRI data

% Average across the 10 repeats
fmritest1_avg = mean(fmritest.test1, 3);  % Average over repeats

% Assuming X_test is defined as the fMRI data
X_test = fmritest1_avg';  % fMRI data (Nt x Nv)

%%%%%%%%%%%%%%% Load trained decoding model weights (W)
subject_id = 'sub1';

dataroot=['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/test1/test1_Ahat_event_time/', subject_id, 'decoding/'];
load(fullfile(dataroot, 'path_to_W.mat'), 'W');

% Reconstruct the feature maps from BOLD data
decoded_features = X_test * W; % Estimated CNN feature maps (Nt x Nu)


%%%%%%%%%%%%%%% Function to apply PredNet to the decoded features

reconstructed_images = cell(length(adjusted_event_indices), 1);
for i = 1:length(adjusted_event_indices)
    feature_map = squeeze(decoded_features(adjusted_event_indices(i), :, :));  % Get feature map at the event time
    reconstructed_images{i} = apply_prednet(bias, kernel, feature_map);  % Apply PredNet to reconstruct the image
end
