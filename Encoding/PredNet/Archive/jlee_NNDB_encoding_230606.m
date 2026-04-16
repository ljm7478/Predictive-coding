%% Loading the data (X - featuremap,  Y - fMRI)
clc; clear

% Directory paths
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/';

% Open feature map
secpath = fullfile(dataroot, 'total', 'pre1_reframes.h5');
X = h5read(secpath, '/data');
X = reshape(X, [], size(X, 4)); % Reshape feature map to 2D: (61440, 136750)

% Open fMRI data
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))
sub1_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-1/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
Y = sub1_cii.cdata; % Transpose fMRI data to match feature map dimensions

%% Temporal alignment
% Temporal resolution of the fMRI data (TR in seconds)
fmri_resolution = 1;

% Temporal resolution of the feature map (frames per second)
feature_resolution = 25;

% Define the HRF
p = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/feature_resolution, p);
hrf = hrf(:);

% Convolve each feature in the feature map with the HRF
convolved_feature_map = zeros(size(X));
for feature_idx = 1:size(X, 2)
    convolved_feature_map(:, feature_idx) = conv(X(:, feature_idx), hrf, 'same');
end

% Downsample the convolved feature map to match the temporal resolution of the fMRI data
resampled_feature_map = resample(convolved_feature_map', size(Y, 2), size(convolved_feature_map, 2))';

%% Split the data into train and test sets
train_percentage = 0.8;  % Percentage of data for training
train_samples = round(size(resampled_feature_map, 1) * train_percentage);

%% Randomly shuffle the data
idx = randperm(size(resampled_feature_map, 1));
X_train = resampled_feature_map(idx(1:train_samples), :);
X_test = resampled_feature_map(idx(train_samples+1:end), :);
Y_train = Y(idx(1:train_samples), :);
Y_test = Y(idx(train_samples+1:end), :);

%% save fMRI into .h5
h5create([saveroot,'sub-1_fMRI.h5'],'/Y_train',...
    [size(Y_train)],'Datatype','single');
h5write([saveroot,'total_500SUM_PredNet_feature_maps_avg.h5'],'/Y_train', Y_train);
%% Ridge regression
lambda = 0.1; % Regularization parameter

% Transpose the matrices
Y_train = Y_train';
X_train = X_train';

% Train the ridge regression model
ridge_model = ridge(Y_train(:), X_train, lambda);

% Predict the fMRI response for the test set using the trained model
Y_test_predicted = [ones(size(X_test, 1), 1) X_test] * ridge_model;

% Calculate the mean squared error (MSE) between predicted and actual fMRI response
mse = mean((Y_test - Y_test_predicted).^2);

%% Visualization
% Select a random voxel index for visualization
voxel_idx = randi(size(Y_test, 2));

% Plot the predicted and actual fMRI responses for the selected voxel
figure
plot(Y_test(:, voxel_idx), 'b', 'LineWidth', 2)
hold on
plot(Y_test_predicted(:, voxel_idx), 'r--', 'LineWidth', 2)
hold off
xlabel('Time')
ylabel('fMRI Response')
title('Actual vs Predicted fMRI Response')
legend('Actual', 'Predicted')

%% Split the data into train and test sets
% train_percentage = 0.8;  % Percentage of data for training
% train_samples = round(size(X_convolved, 2) * train_percentage);
% 
% % Randomly shuffle the data
% idx = randperm(size(X_convolved, 2));
% 
% X_train = X_convolved(:, idx(1:train_samples));
% X_test = X_convolved(:, idx(train_samples+1:end));
% 
% Y_train = Y(:, idx(1:train_samples));
% Y_test = Y(:, idx(train_samples+1:end));



%% Step 1: Loading the data
% clc; clear
% % Directory paths
% dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
% saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/';
% 
% % Open featuremap (X)
% secpath = fullfile(dataroot, 'total', 'pre1_reframes.h5');
% X = h5read(secpath, '/data');
% dim = size(X);
% X = reshape(X, prod(dim(1:end-1)), dim(end));
% 
% %% HRF
% % Define the HRF parameters
% hrf_duration_tr = 15;  % HRF duration in TRs
% hrf_tr = 1;  % HRF time resolution in seconds (should match fMRI TR)
% hrf_samples = hrf_duration_tr;  % Number of HRF samples (since hrf_tr = 1)
% hrf_fps = 1 / hrf_tr;  % HRF sampling rate in Hz
% 
% % Generate the HRF
% hrf = spm_hrf(hrf_tr, hrf_fps);
% 
% % Resample the HRF to match the movie frame rate
% movie_fps = 25;
% hrf_resampled = resample(hrf, movie_fps, hrf_fps);
% 
% %% Convolve each feature map with the HRF
% X_convolved = zeros(size(X));
% 
% for i = 1:size(X, 2)
%     X_convolved(:, i) = conv(X(:, i), hrf_resampled, 'same');
% end
% 
% % Split the convolved data into train and test sets if desired
% train_indices = 1:round(size(X_convolved, 1) * 0.8);  % 80% for training
% test_indices = round(size(X_convolved, 1) * 0.8) + 1:size(X_convolved, 1);  % 20% for testing
% 
% X_train = X_convolved(train_indices, :);
% X_test = X_convolved(test_indices, :);
% 
%% Step 1: Loading the data
% clc; clear
% % Directory paths
% dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
% saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/feature_processed_encoding/';
% 
% % Open featuremap (X)
% secpath = fullfile(dataroot, 'total', 'pre1_reframes.h5');
% X = h5read(secpath, '/data');
% dim = size(X);
% X = reshape(X, prod(dim(1:end-1)), dim(end));
% 
% %% Step 1.2: Splitting the data for X
% % Calculate the number of rows per segment
% rowsPerSegment = size(X, 1) / 6;
% 
% % Generate indices to divide the rows into segments
% rowIndices = round(linspace(1, size(X, 1) + 1, 7));
% 
% % Create a cell array to store the segments
% segments = cell(1, 6);
% 
% % Split the array into segments
% for i = 1:6
%     segmentRows = rowIndices(i):rowIndices(i + 1) - 1;
%     segments{i} = X(segmentRows, :);
%     name = ['X_segments', num2str(i)];
%     eval([name, ' = segments{i};']);
% end
% 
% %% HRF
% % Define the HRF parameters
% hrf_duration_tr = 15;  % HRF duration in TRs
% hrf_tr = 1;  % HRF time resolution in seconds (should match fMRI TR)
% hrf_samples = hrf_duration_tr;  % Number of HRF samples (since hrf_tr = 1)
% hrf_fps = 1 / hrf_tr;  % HRF sampling rate in Hz
% 
% % Generate the HRF
% hrf = spm_hrf(hrf_tr, hrf_fps);
% 
% % Resample the HRF to match the movie frame rate
% movie_fps = 25;
% hrf_resampled = resample(hrf, movie_fps, hrf_fps);
% 
% % Convolve each feature map with the HRF
% X_convolved = zeros(size(X));
% for i = 1:size(X, 2)
%     X_convolved(:, i) = conv(X(:, i), hrf_resampled, 'same');
% end
% 
% % Save the convolved feature maps
% savepath = fullfile(saveroot, 'convolved_feature_maps.mat');
% save(savepath, 'X_convolved');
%% Step 2: Load fmri responses (Y) -NNDB
disp('Load fmri responses -NNDB')
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))

sub1_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-1/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub2_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-2/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub3_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-3/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub4_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-4/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
sub5_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-5/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');

%% Step 2.2: Splitting the data for Y
% Assume you have already split the data into training and testing sets: X_train, X_test, y_train, y_test

% Calculate the number of rows per segment
rowsPerSegment = dim(1) / 6;

% Generate indices to divide the rows into segments
rowIndices = round(linspace(1, dim(1) + 1, 7));

% Create a cell array to store the segments
segments = cell(1, 6);

% Split the array into segments
for i = 1:6
    segmentRows = rowIndices(i):rowIndices(i+1)-1;
    segments{i} = X(segmentRows, :);
    name = strcat('X_segments',num2str(i));
    eval([name, '= segments{i};']);
end
%% Step 3: Training the encoding model
model = fitlm(X_train, y_train);

%% Step 4: Predicting the responses
y_pred = predict(model, X_test);

%% Step 5: Evaluating the predictions
correlation = corr(y_pred, y_test);
rmse = sqrt(mean((y_test - y_pred).^2));
explained_variance = 1 - var(y_test - y_pred) / var(y_test);

%% Step 6: Visualizing the correlation
scatter(y_test, y_pred);
xlabel('Actual fMRI responses');
ylabel('Predicted fMRI responses');
title('Correlation between actual and predicted fMRI responses');

%% Step 7: Analyzing the results
disp(['Correlation coefficient: ' num2str(correlation)]);
disp(['RMSE: ' num2str(rmse)]);
disp(['Explained variance: ' num2str(explained_variance)]);
