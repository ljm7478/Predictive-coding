%% separate fMRI into test and train for Jack Gallant encoding code
clc; clear

% Open fMRI data
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'))
sub1_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-1/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
Y = sub1_cii.cdata'; % Transpose fMRI data to match feature map dimensions
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/data_for_JGallant_encoding/';

% Define the ratio for train and test set
trainRatio = 0.7;

% Get the total number of samples
totalSamples = size(Y, 1);

% Create a partition object for cross-validation
cv = cvpartition(totalSamples, 'HoldOut', 1 - trainRatio);

% Get the indices of the training and test set
trainIndices = training(cv);
testIndices = test(cv);

% Separate the data into training and test sets
trainData = Y(trainIndices, :);
testData = Y(testIndices, :);

%to fit data order with python code 
trainData=trainData';
testData=testData';

% Set the file name
filename = [saveroot, 'sub-1_responses.h5'];

% Create the HDF5 file
h5create(filename, '/Y_train', size(trainData));
h5create(filename, '/Y_test', size(testData));

% Write trainData to the HDF5 file
h5write(filename, '/Y_train', trainData);

% Write testData to the HDF5 file
h5write(filename, '/Y_test', testData);


%% separate feature into test and train for Jack Gallant encoding code
clc; clear

% Directory paths
dataroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/raw_feature_maps';
saveroot = '/local_raid2/03_user/jungmin/01_Encoding/01_HWen_Encoding/02_Goal-Driven-DL/01_NeuralEncodingDecoding/stimuli/PredNet/02_FG_pretrained_test_500SUM_PredNet/data_for_JGallant_encoding/';

% Open feature map
secpath = fullfile(dataroot, 'total', 'pre1_reframes.h5');
X = h5read(secpath, '/data');
X = reshape(X, [], size(X, 4)); % Reshape feature map to 2D: (61440, 136750)

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

%load Y
sub1_cii = ciftiopen('/zeus-share/local_raid2/user/sunghyoung/01_project/04_NNDB/ciftify_archive/ciftify/sub-1/MNINonLinear/Results/task-500daysofsummer/task-500daysofsummer_Atlas_s0.dtseries.nii','wb_command');
Y = sub1_cii.cdata; % Transpose fMRI data to match feature map dimensions

% Downsample the convolved feature map to match the temporal resolution of the fMRI data
resampled_feature_map = resample(convolved_feature_map', size(Y, 2), size(convolved_feature_map, 2))';
resampled_feature_map=resampled_feature_map';
clear X;

% Define the ratio for train and test set
trainRatio = 0.7;

% Get the total number of samples
totalSamples = size(resampled_feature_map, 1);

% Create a partition object for cross-validation
cv = cvpartition(totalSamples, 'HoldOut', 1 - trainRatio);

% Get the indices of the training and test set
trainIndices = training(cv);
testIndices = test(cv);

% Separate the data into training and test sets
trainData = resampled_feature_map(trainIndices, :);
testData = resampled_feature_map(testIndices, :);

%to fit data order with python code 
trainData=trainData';
testData=testData';

% Set the file name
filename = [saveroot, 'prednet_feature.h5'];

% Create the HDF5 file
h5create(filename, '/X_train', size(trainData));
h5create(filename, '/X_test', size(testData));

% Write trainData to the HDF5 file
h5write(filename, '/X_train', trainData);

% Write testData to the HDF5 file
h5write(filename, '/X_test', testData);