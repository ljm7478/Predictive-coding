%% fMRI Preprocessing Script for Budapest Dataset
% Only: mask subcortical + z-score (denoising already done)

clc; clear;

addpath(genpath('/local_raid1/202511_SW/HCPpipelines/4.8.0/global/matlab/cifti-matlab'));
clc;
%% --- Setup ---
Nv = 59412; % Cortical vertices
num_runs = 5;

% Budapest paths
fmripath_template = '/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/%s/%s/MNINonLinear/Results/task-movie_run-%02d';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/fmri_new';

subjects = {'sub-sid000005', 'sub-sid000007', 'sub-sid000009', 'sub-sid000010', 'sub-sid000013', ...
            'sub-sid000020', 'sub-sid000021', 'sub-sid000024', 'sub-sid000025', 'sub-sid000029', ...
            'sub-sid000030', 'sub-sid000034', 'sub-sid000050', 'sub-sid000052', 'sub-sid000055', ...
            'sub-sid000114', 'sub-sid000120', 'sub-sid000134', 'sub-sid000142', 'sub-sid000278', ...
            'sub-sid000416', 'sub-sid000499', 'sub-sid000522', 'sub-sid000535', 'sub-sid000560'};

%% --- Main Processing Loop ---
for subj_idx = 1:length(subjects)
    subject = subjects{subj_idx};
    fprintf('\n=== Subject: %s (%d/%d) ===\n', subject, subj_idx, length(subjects));

    fmri_data_concat = []; 
    fmri_data_per_run = struct();

    for run_idx = 1:num_runs
        fprintf('  Run %d...', run_idx);
        
        % Load CIFTI
        fmripath = sprintf(fmripath_template, subject, subject, run_idx);
        filename = sprintf('/task-movie_run-%02d_Atlas_s3.dtseries.nii', run_idx);
        fmri_file = fullfile(fmripath, filename);
        
        if ~exist(fmri_file, 'file')
            fprintf(' NOT FOUND, skipping\n');
            continue;
        end
        
        cii = ciftiopen(fmri_file, 'wb_command');
        
        % Mask subcortical (keep only cortical vertices)
        data = double(cii.cdata(1:Nv, :));
        fprintf(' %d TRs, ', size(data, 2));
        
        % Z-score only (detrending already done in bash script)
        data = bsxfun(@minus, data, mean(data, 2));
        sigma = std(data, 0, 2);
        sigma(sigma < 1e-6) = 1;
        data = bsxfun(@rdivide, data, sigma);

        % Store
        if ~any(isnan(data(:)))
            fmri_data_per_run.(['run', num2str(run_idx)]) = single(data);
            fmri_data_concat = [fmri_data_concat, single(data)];
            fprintf('OK\n');
        else
            fprintf('NaN found, skipping\n');
        end
    end

    % Save
    if ~isempty(fmri_data_concat)
        sub_fmri_dir = fullfile(saveroot, subject);
        if ~exist(sub_fmri_dir, 'dir'), mkdir(sub_fmri_dir); end
        
        save(fullfile(sub_fmri_dir, [subject, '_per_run_fmri.mat']), 'fmri_data_per_run', '-v7.3');
        save(fullfile(sub_fmri_dir, [subject, '_concat_fmri.mat']), 'fmri_data_concat', '-v7.3');
        fprintf('  Saved: %d total TRs\n', size(fmri_data_concat, 2));
    end
end

fprintf('\nDone.\n');