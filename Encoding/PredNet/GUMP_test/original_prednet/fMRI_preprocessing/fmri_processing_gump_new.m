
%% fMRI Preprocessing Script for Voxel-wise Encoding Analysis
%
% This script performs the following standard preprocessing steps for each subject:
% 1. Loads CIFTI dtseries data for each of the 8 movie runs.
% 2. Masks out subcortical vertices to isolate cortical surface data.
% 3. Applies a 4th-order polynomial detrending to each voxel's time series to remove scanner drift.
% 4. Standardizes each voxel's time series (z-scoring) to have a mean of 0 and a standard deviation of 1.
% 5. Saves the processed data for each run individually and as a single concatenated file for each subject.

clc; clear;

%% --- Setup: Paths and Constants ---
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01 _software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

% Define constants
Nv = 59412; % Number of vertices on the cortical surface
num_volumes = [451, 441, 438, 488, 462, 439, 542, 338]; % Number of TRs for each run

% Define file path templates
fmripath_template = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-%d';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new';
  
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', ...
            'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', ...
            'sub-18', 'sub-19', 'sub-20'};

%% --- Main Processing Loop ---
for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};
    fprintf('\n=================================================\n');
    fprintf('Starting preprocessing for subject: %s\n', subject);
    fprintf('=================================================\n');

    % CRITICAL FIX: Initialize concatenated data for EACH subject inside the loop.
    fmri_data_concat = []; 
    fmri_data_per_run = struct();

    for run_idx = 1:8
        fprintf('  Processing Run %d...\n', run_idx);
        
        % 1. Load Data
        fmripath = sprintf(fmripath_template, subject, subject, subject, run_idx);
        filename = sprintf('/ses-movie_task-movie_run-%d_Atlas_s3.dtseries.nii', run_idx);
        fmri_file = fullfile(fmripath, filename);
        
        if ~exist(fmri_file, 'file')
            fprintf('    Warning: File not found, skipping: %s\n', fmri_file);
            continue;
        end
        
        cii = ciftiopen(fmri_file, 'wb_command');
        Nt = num_volumes(run_idx);
        
        % 2. Mask out subcortical vertices
        cortical_mask = false(size(cii.cdata, 1), 1);
        cortical_mask(1:Nv) = true;
        data = double(cii.cdata(cortical_mask, 1:Nt));
        
        % 3. Detrending
        % This removes low-frequency scanner drift using a 4th order polynomial.
        fprintf('    Detrending %d vertices...\n', size(data, 1));
        for i = 1:size(data, 1)
            data(i,:) = amri_sig_detrend(data(i,:), 4);
        end
        
        % 4. Standardization (Voxel-wise Z-scoring)
        % This normalizes each voxel's time series to have mean=0 and std=1.
        data = bsxfun(@minus, data, mean(data, 2));
        sigma = std(data, 0, 2); % Calculate std dev for each voxel across time
        sigma(sigma < 1e-6) = 1; % Prevent division by zero for near-constant voxels
        data = bsxfun(@rdivide, data, sigma);

        % Check for NaN values before storing
        if ~any(isnan(data(:)))
            fmri_data_per_run.(['run', num2str(run_idx)]) = single(data); % Use single for memory efficiency
            fmri_data_concat = [fmri_data_concat, single(data)];
        else
            fprintf('    Warning: NaN values found in Run %d. Skipping storage for this run.\n', run_idx);
        end
    end

    % 5. Save Processed Data for the Subject
    if ~isempty(fmri_data_concat)
        sub_fmri_dir = fullfile(saveroot, subject);
        if ~exist(sub_fmri_dir, 'dir')
            mkdir(sub_fmri_dir);
        end
        
        fprintf('\n  Saving processed data for subject %s...\n', subject);
        save(fullfile(sub_fmri_dir, [subject, '_per_run_fmri.mat']), 'fmri_data_per_run', '-v7.3');
        save(fullfile(sub_fmri_dir, [subject, '_concat_fmri.mat']), 'fmri_data_concat', '-v7.3');
        fprintf('  Data saved successfully.\n');
    else
        fprintf('\n  Warning: No valid data processed for subject %s. Nothing saved.\n', subject);
    end
end

fprintf('\nAll subjects processed.\n');
