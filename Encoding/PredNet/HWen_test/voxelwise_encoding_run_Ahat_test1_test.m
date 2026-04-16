clc; clear
%% Load test fMRI responses
subject_id = 'subject1';
dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/', subject_id, '_fmri/'];

load([dataroot,'testing_fmri.mat'],'fmritest'); % Load preprocessed fMRI data

% For test1, disregard the first volume and the last 4 volumes, and average across the 10 repeats
fmritest1_avg = mean(fmritest.test1, 3);

%% Predict BOLD responses using the whole time series
layername = {'/Ahat1', '/Ahat2', '/Ahat3', '/Ahat4'};  
subject_id = 'sub1';

% %modified-32c
dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/Ahat/';
saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_3_32_64_96_192channel/test1/test1_Ahat_event_time/', subject_id, '/'];

% Path to the modified-96c data
% dataroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/layer1_4/';
% saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/test1/test1_Ahat_event_time/', subject_id, '/'];

num_layers = length(layername);
all_pred_fmri = cell(num_layers, 1); 

for lay = 1:num_layers
    disp(['Layer: ', num2str(lay)]);
    secpath = fullfile(dataroot, 'PredNet_feature_maps_pcareduced_test1.h5');
    if exist(secpath, 'file')
        Y_test = h5read(secpath, [layername{lay}, '/data']);
    else
        disp(['File does not exist: ', secpath]);
        continue;
    end
    
    % Load weights matrix
    W_filename = fullfile(dataroot, subject_id, ['W_lambda5_', layername{lay}(2:end), '.mat']);
    load(W_filename, 'W');

    % Predict BOLD signal for the whole time series
    pred_fmri = Y_test * W;

    % Store predicted BOLD signals for this layer
    all_pred_fmri{lay} = pred_fmri;
end

%% Define event times and extract corresponding signals
% Define your event times in seconds
test1_event_times = [13, 30, 45, 57, 80, 92, 105, 117, 127, 138, 152, 167, 184, 195, ...
                     208, 223, 233, 242, 258, 283, 297, 309, 330, 336, 345, 357, 369, ...
                     386, 398, 410, 427, 434, 446, 462];

% Adjust event times to match the HRF peak delay and convert to TR indices
hrf_peak_delay = 4; 
TR = 2.0;
adjusted_event_times = test1_event_times + hrf_peak_delay;
adjusted_event_indices = round(adjusted_event_times / TR);

% Ensure the indices are within the valid range [1, 240]
adjusted_event_indices = adjusted_event_indices(adjusted_event_indices > 0 & adjusted_event_indices <= 240);

% Extract the predicted BOLD responses at the event times
extracted_pred_fmri = cell(num_layers, 1);
for lay = 1:num_layers
    pred_fmri = all_pred_fmri{lay};
    extracted_pred_fmri{lay} = pred_fmri(adjusted_event_indices, :);
end

% Extract actual fMRI responses at the event times
extracted_fmritest1_avg = fmritest1_avg(:, adjusted_event_indices)';

%% Correlate extracted signals
all_corr = cell(num_layers, 1);
for lay = 1:num_layers
    pred_fmri_events = extracted_pred_fmri{lay};
    correlation_scores = zeros(1, size(pred_fmri_events, 2));
    
    for v = 1:size(pred_fmri_events, 2)
        correlation_scores(v) = corr(pred_fmri_events(:, v), extracted_fmritest1_avg(:, v));
    end
    
    all_corr{lay} = correlation_scores;

    % Save the correlation results
    file_name = fullfile(saveroot, [subject_id, '_corr_lambda5_', layername{lay}(2:end), '.mat']);
    save(file_name, 'correlation_scores', '-v7.3');
end

% Concatenate correlation results from all layers for visualization
concat_corr = cat(1, all_corr{:});

%% Save and visualize the correlation results in CIFTI format
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Load CIFTI structure 
subject_id = 'subject1';
fmripath = ['/combinelab/03_user/jungmin/02_data/08_NeuralEncodingDecoding/', subject_id, '/video_fmri_dataset/', subject_id, '/fmri/'];
filename = ['seg1/cifti/seg1_1_Atlas.dtseries.nii'];
cii = ciftiopen(fullfile(fmripath, filename), 'wb_command');

% Save back into dtseries for visualization
cii.cdata = concat_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);
cii.diminfo{1, 1}.models(3:end) = [];
save_file_name = fullfile(saveroot, [subject_id, '_lambda5_Ahat_rev.dtseries.nii']);
ciftisavereset(cii, save_file_name, 'wb_command');

disp(['Saved results to: ', save_file_name]);
