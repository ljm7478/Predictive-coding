clc; clear

%% Paths and Setup
subject_id = 'sub1';

% Paths for the modified-96c encoding analysis
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/';

% WH time
% E_saveroot = fullfile(dataroot, 'test1/test1_E_WH_time/', subject_id, '/');
% P_saveroot = fullfile(dataroot, 'test1/test1_Ahat_WH_time/', subject_id, '/');

% WH time
E_saveroot = fullfile(dataroot, 'test1/test1_E_event_time/', subject_id, '/');
P_saveroot = fullfile(dataroot, 'test1/test1_Ahat_event_time/', subject_id, '/');


% Load encoding results for Error feature
layername_error = {'/E1', '/E2', '/E3', '/E4'};
num_layers = length(layername_error);

E_all_corr = cell(num_layers, 1);
for lay = 1:num_layers
    file_name = fullfile(E_saveroot, [subject_id, '_corr_lambda5_', layername_error{lay}(2:end), '.mat']);
    temp = load(file_name, "correlation_scores");
    E_all_corr{lay} = temp.correlation_scores;
end

% Concatenate all layers for Error
E_concat_corr = cat(1, E_all_corr{:});

% Load encoding results for Prediction feature
layername_pred = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};

P_all_corr = cell(num_layers, 1);
for lay = 1:num_layers
    file_name = fullfile(P_saveroot, [subject_id, '_corr_lambda5_', layername_pred{lay}(2:end), '.mat']);
    temp = load(file_name, "correlation_scores");
    P_all_corr{lay} = temp.correlation_scores;
end

% Concatenate all layers for Prediction
P_concat_corr = cat(1, P_all_corr{:});

%% Compute the Dominance Map (Prediction - Error)
dominance_map = P_concat_corr - E_concat_corr;

%% Statistical Comparison (Optional)
% If you want to perform a statistical test, you need repeated measures.
% Here, we assume you have access to such data across multiple subjects or runs.
% Adjust this as needed based on your actual data structure.
% [~, p_values, ~, stats] = ttest(P_concat_corr, E_concat_corr, 'Dim', 2);

% Threshold the difference map to highlight dominant regions
threshold = 0.1;  % Threshold for visualization
dominance_map(abs(dominance_map) < threshold) = 0;  % Set values below threshold to zero

%% Binarize the map: 1 if Error is dominant, 2 if Prediction is dominant
binarized_map = zeros(size(dominance_map));  % Initialize with zeros

binarized_map(dominance_map < 0) = 1;  % Error is dominant
binarized_map(dominance_map > 0) = 2;  % Prediction is dominant

% Optional: Handle ties (if dominance_map == 0)
% binarized_map(dominance_map == 0) = 0;  % 0 could represent no dominance

%% Load CIFTI Template for Saving the Dominance Map
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Load a template CIFTI file
subject_id = 'subject1';
fmripath = ['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/'];
seg = 1;
filename = ['seg', num2str(seg), '/cifti/seg', num2str(seg), '_1_Atlas.dtseries.nii'];
cii = ciftiopen(fullfile(fmripath, filename), 'wb_command');

%% Replace the CIFTI Data with the Dominance Map
cii.cdata = binarized_map';

% Update dimensions to match the new data
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];

%% Save the Updated CIFTI File
saveroot=fullfile(dataroot,'test1/')
save_file_name = fullfile(saveroot, [subject_id, '_dominance_Prediction_vs_Error_event_time.dtseries.nii']);
ciftisavereset(cii, save_file_name, 'wb_command');

disp(['Saved dominance map to: ', save_file_name]);
