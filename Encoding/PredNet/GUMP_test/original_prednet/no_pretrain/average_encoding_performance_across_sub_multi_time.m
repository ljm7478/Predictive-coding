clc; clear;

% List of subject IDs
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', ...
                'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', ...
                'sub-18', 'sub-19', 'sub-20'};

% Define layer names (4 layers in this example)
% Prediction feature
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6'};

% Prediction Error feature
% layername = {'/E0'; '/E1'; '/E2'; '/E3'};
% layername = {'/E0'; '/E1'; '/E2'; '/E3'; '/E4'; '/E5'; '/E6'};

% long-term memory feature
% layername = {'/C0'; '/C1'; '/C2'; '/C3'};
% layername = {'/C0'; '/C1'; '/C2'; '/C3'; '/C4'; '/C5'; '/C6'};

% layer info
if length(layername) == 4
    layer_name = 'layer4';
elseif length(layername) == 7
    layer_name = 'layer7';
end

% Define feature types to process
types = {'Ahat', 'E', 'C'};
% types = {'C'};

% Add required paths for CIFTI processing
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Load a reference CIFTI structure to use as a template
fmripath = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/';
filename = 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';
cii = ciftiopen([fmripath, filename],'wb_command');

% Loop through both Ahat and E feature types
for t = 1 %length(types)
    type = types{t};
    disp(['Processing type: ', type]);
    
    % Define the root directory where each type's data is stored
    dataroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/multi_time_result/2second_50frame/', ...
                layer_name, '/', type, '/'];

    % Initialize and load each subject's average correlation map
    for sub = 1:length(num_subjects)
        subject_id = num_subjects{sub};
        disp(['  current sub is ', subject_id]);

        file_name = [subject_id, '_avgcorr_lambda5.mat'];
        % file_name = [subject_id, '_lagged_avgcorr_lambda5.mat'];

        load(fullfile(dataroot, subject_id, file_name), 'concat_corr');

        % Initialize storage on first subject
        if sub == 1
            all_subject_corr = zeros(length(num_subjects), size(concat_corr, 1), size(concat_corr, 2), 'single');
        end
        all_subject_corr(sub, :, :) = concat_corr;
    end

    % Compute group average across subjects
    group_avg_corr = squeeze(mean(all_subject_corr, 1)); % [voxel x layer]

    % Prepare the CIFTI structure with new averaged data
    new_cii = cii;
    new_cii.cdata = group_avg_corr';  % [layer x voxel]
    new_cii.diminfo{1,1}.length = size(new_cii.cdata, 1);  % Number of layers
    new_cii.diminfo{1,2}.length = size(new_cii.cdata, 2);  % Number of voxels
    new_cii.diminfo{1,1}.models(3:end) = [];  % Clean extra models if any

    % Save the averaged correlation map as a CIFTI file
    save_file_name = fullfile(dataroot, [type '_avg_lambda5.dtseries.nii']);
    ciftisavereset(new_cii, save_file_name, 'wb_command');
end
