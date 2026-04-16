clc;clear
% first layer of the model
subject_id = 'sub1';

%dir
% %modified-32c
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/Ahat/';
saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/test1/test1_Ahat_event_time/', subject_id, '/'];

% Path to the modified-96c data
% dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/layer1_4/';
% saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/test1/test1_Ahat_event_time/', subject_id, '/'];
%% Load training fmri responses
subject_id = 'subject1';
dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];
load([dataroot,'training_fmri.mat'],'fmri'); % from movie_fmri_processing.m
fmri_avg = (fmri.data1+fmri.data2) / 2; % average across repeats
fmri_avg = reshape(fmri_avg, size(fmri_avg,1), size(fmri_avg,2)*size(fmri_avg,3));
%% Load test fMRI responses
subject_id = 'subject1';

dataroot_fmri = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];
load([dataroot_fmri, 'testing_fmri.mat'], 'fmritest');  % Load preprocessed fMRI data

% For test1, disregard the first volume and the last 4 volumes (already processed & saved in fmritest), and average across the 10 repeats
fmritest1_avg = mean(fmritest.test1, 3);  % Average over repeats

%% Load test feature map (CNN activations)
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/Ahat/';
layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};
lay = 1;

secpath = [dataroot,'PredNet_feature_maps_pcareduced_concatenated.h5'];
if exist(secpath, 'file')
    Y_train = h5read(secpath, [layername{lay}, '/data']);
else
    disp(['File does not exist: ', secpath]);
end

%% Decoding (Reconstruction) train with training movie
clc;
subject_id = 'sub1';
saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/test1/test1_Ahat_event_time/', subject_id, '/', 'decoding', '/'];

% Set up paths and data
X = fmri_avg';  % fMRI data (Nt x Nv)
Y = Y_train;  % Using first layer as an example (Nt x Nu)

% Check if dimensions match
assert(size(X, 1) == size(Y, 1), 'Mismatch in time points between fMRI data and feature maps.');

% Define lambda (regularization parameters) and nfold (cross-validation folds)
% lambda = logspace(-4, 2, 50);
lambda = [0.1:0.2:0.9];   % same lambda as encoding
nfold = 20; % from paper

%%%%% orig parameter setting
% % Set up opts structure
% opts = struct();
% opts.InitialMomentum = 0.5;
% opts.FinalMomentum = 0.9;
% opts.InitialMomentumIter = 20;
% opts.MaxIter = 50;
% opts.DropOutRate1 = 0.3;
% opts.DropOutRate2 = 0; 
% opts.StepRatio = 5e-4;
% opts.MinStepRatio = 1e-4;
% opts.SparsityCost = 0;
% opts.lambda = 0.05;
% opts.BatchSize = 100;
% opts.disp_flag = 1;
% opts.IterNum = 5;  % Save model every IterNum iterations
% opts.filedir = saveroot(1:end-1);  %jmlee

%%%%% jmlee parameter setting
opts = struct();
opts.InitialMomentum = 0.5;
opts.FinalMomentum = 0.9;
opts.InitialMomentumIter = 2;  % Adjust if you want fewer momentum-adjustment iterations
opts.MaxIter = 5;  % 5 epochs
opts.DropOutRate1 = 0.3;
opts.DropOutRate2 = 0; 
opts.StepRatio = 5e-4;
opts.MinStepRatio = 1e-4;
opts.SparsityCost = 0;
opts.lambda = 0.05;
opts.BatchSize = 100;  % Keep batch size the same
opts.disp_flag = 1;
opts.IterNum = 1;  % Save model every iteration, adjusted for the smaller number of iterations
opts.filedir = saveroot(1:end-1);

% Run visual reconstruction
[W, Rmat, Errmat, Lambda] = visual_reconstruction(Y, X, lambda, nfold, opts);

% save W (feature x voxel)
file_name = ['W_', layername{lay}(2:end) , '.mat']
save([saveroot, file_name], 'W', '-v7.3');

%% test decoding model with one of the testing movies
X = fmritest1_avg';  % fMRI data (Nt x Nv)

% Reconstruct the feature maps from BOLD data
reconstructed_Y = X * W;  % Estimated CNN feature maps (Nt x Nu)
file_name = ['reconstructed_Y_', layername{lay}(2:end) , '.mat']
save([saveroot, file_name], 'reconstructed_Y', '-v7.3');
%% Visualization
% Define your event times in seconds
test1_event_times = [13, 30, 45, 57, 80, 92, 105, 117, 127, 138, 152, 167, 184, 195, ...
                     208, 223, 233, 242, 258, 283, 297, 309, 330, 336, 345, 357, 369, ...
                     386, 398, 410, 427, 434, 446, 462];

% Adjust event times to match the HRF peak delay and convert to TR indices
hrf_peak_delay = 4;  % seconds
TR = 2.0;  % seconds
adjusted_event_times = test1_event_times + hrf_peak_delay;
adjusted_event_indices = round(adjusted_event_times / TR);

% Ensure the indices are within the valid range [1, 240]
adjusted_event_indices = adjusted_event_indices(adjusted_event_indices > 0 & adjusted_event_indices <= 240);


figure;
subplot(1, 2, 1);
plot(Y(time_point, :));  % Plot the PCA-reduced feature map
title('Original PCA-Reduced Feature Map');

subplot(1, 2, 2);
plot(reconstructed_Y(time_point, :));  % Plot the PCA-reconstructed feature map
title('Reconstructed PCA-Reduced Feature Map');




% Set dimensions for reshaping
height = 64;
width = 80;
num_channels = 32;

size_Y = size(Y(time_point, :));
expected_size = height * width * num_channels;

if size_Y(2) ~= expected_size
    error('The dimensions of the feature map do not match the expected size. Check height, width, and num_channels.');
else
    % Proceed with reshaping if the sizes match
    original_map = reshape(Y(time_point, :), [height, width, num_channels]);  % Original feature map
    reconstructed_map = reshape(reconstructed_Y(time_point, :), [height, width, num_channels]);  % Reconstructed feature map
    
    % Visualize the first channel of the feature maps
    figure;
    subplot(1, 2, 1);
    imshow(original_map(:, :, 1), []);  % Display the first channel of the original feature map
    title('Original Feature Map - Channel 1');
    
    subplot(1, 2, 2);
    imshow(reconstructed_map(:, :, 1), []);  % Display the first channel of the reconstructed feature map
    title('Reconstructed Feature Map - Channel 1');
end

% % Select the first time point for visualization
% time_point = adjusted_event_indices(1);
% 
% % Reshape the original and reconstructed maps
% original_map = reshape(Y(time_point, :), [height, width, num_channels]);  % Original feature map
% reconstructed_map = reshape(reconstructed_Y(time_point, :), [height, width, num_channels]);  % Reconstructed feature map
% 
% % Visualize the first channel of the feature maps
% figure;
% subplot(1, 2, 1);
% imshow(original_map(:, :, 1), []);  % Display the first channel of the original feature map
% title('Original Feature Map - Channel 1');
% 
% subplot(1, 2, 2);
% imshow(reconstructed_map(:, :, 1), []);  % Display the first channel of the reconstructed feature map
% title('Reconstructed Feature Map - Channel 1');

% % Example visualization for one time point
% height=64;
% width=80;
% num_channels=32;
% time_point = adjusted_event_indices(1);
% original_map = reshape(Y(time_point, :), [height, width, num_channels]);  % Adjust dimensions as needed
% reconstructed_map = reshape(reconstructed_Y(time_point, :), [height, width, num_channels]);  % Adjust dimensions as needed
% 
% figure;
% subplot(1,2,1); imshow(original_map); title('Original Feature Map');
% subplot(1,2,2); imshow(reconstructed_map); title('Reconstructed Feature Map');