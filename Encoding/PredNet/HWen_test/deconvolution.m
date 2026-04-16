%% decoding 
clc;clear

%% load PredNet weight of 1st layer
% HDF5 file containing PredNet weights
hdf5_file = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/weights/Titanic_train_250epoch_layers5_96channel_alexnet_follow.hdf5';

% Inspect the file to understand its structure
info = h5info(hdf5_file);
disp(info);  % This will print the structure of your HDF5 file, including datasets and groups

% Replace with the actual path derived from the output
bias_path = '/model_weights/pred_net_1/pred_net_1/layer_ahat_1/bias:0';
kernel_path = '/model_weights/pred_net_1/pred_net_1/layer_ahat_1/kernel:0';

% Extract the bias and kernel
bias = h5read(hdf5_file, bias_path);
kernel = h5read(hdf5_file, kernel_path);

% Display the sizes to ensure correct extraction
disp(size(bias));   % This should show the dimensions of the bias
disp(size(kernel)); % This should show the dimensions of the kernel

% Save the extracted weights and biases into a .mat file for later use
% savedir='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/deconvolution/';
% save(fullfile(savedir,'ahat1_weights.mat'), 'bias', 'kernel');

% Load the saved weights and biases
load(fullfile(savedir,'ahat1_weights.mat'), 'bias', 'kernel');

%% Deconvolution begin with decoding model W

%%%%%%%%%%%%%%% Load test fMRI responses
subject_id = 'subject1';

dataroot_fmri = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];
load([dataroot_fmri, 'testing_fmri.mat'], 'fmritest');  % Load preprocessed fMRI data

% For test1, disregard the first volume and the last 4 volumes (already processed & saved in fmritest), and average across the 10 repeats
fmritest1_avg = mean(fmritest.test1, 3);  % Average over repeats

%%%%%%%%%%%%%%% Reconstruct the feature maps from BOLD data
decoded_features = X_test * W';  % Estimated CNN feature maps (Nt x Nu)


%%%%%%%%%%%%%%% Function to apply PredNet to the decoded features
function reconstructed_image = apply_prednet(prednet_model, decoded_features)
    % Apply PredNet to the decoded features to reconstruct the image
    % In practice, this would involve running the decoded features through
    % the PredNet architecture to generate the final predicted frame.

    reconstructed_image = prednet_forward(prednet_model, decoded_features);
end

% Example usage:
reconstructed_images = cell(length(adjusted_event_indices), 1);
for i = 1:length(adjusted_event_indices)
    feature_map = squeeze(decoded_features(adjusted_event_indices(i), :, :));  % Get feature map at the event time
    reconstructed_images{i} = apply_prednet(prednet_model, feature_map);  % Apply PredNet to reconstruct the image
end
