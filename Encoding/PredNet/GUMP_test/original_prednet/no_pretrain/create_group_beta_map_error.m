%% Create Group-Average Beta Maps from Individual Subject Encoding Weights (W)
%
% This script implements the post-encoding analysis pipeline:
% 1. For each subject and each layer, it loads the saved encoding weights (W matrix).
% 2. It performs PCA on the W matrix ([Features x Voxels]) to find the dominant
%    pattern of voxel weights across features.
% 3. It extracts the first principal component (PC1) loading vector for each subject.
% 4. It averages these PC1 vectors across all subjects to create a group-level beta map for each layer.
% 5. It saves the final group-level beta maps for all layers into a single .dtseries.nii file.
%
% This should be run AFTER 'run_encoding_pipeline.m' has completed for all subjects.

clc; clear;
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
clc;

%% --- Configuration ---
% Define the feature types you want to process in this run.
FEATURE_TYPES_TO_PROCESS = {'Ahat', 'E'};

% --- Paths ---
BASE_SAVE_ROOT = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/no-pretrain/result/layer4/';
CIFTI_TEMPLATE_FILE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';

% --- Analysis Parameters ---
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
num_layers = 3;

% Add CIFTI tools to path
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

%% --- Main Group Averaging Loop ---
cii_template = ciftiopen(CIFTI_TEMPLATE_FILE, 'wb_command');

for t = 2 %:length(FEATURE_TYPES_TO_PROCESS)
    feature_type = FEATURE_TYPES_TO_PROCESS{t};
    fprintf('\n=================================================\n');
    fprintf('Processing Group Beta Maps for Feature Type: %s\n', feature_type);
    fprintf('=================================================\n');
    
    dataroot = fullfile(BASE_SAVE_ROOT, feature_type);
    num_voxels = 59412;
    layer_wise_group_beta = nan(num_layers, num_voxels, 'single');

    for lay = 0:num_layers
        fprintf('  Aggregating beta maps for layer %d...\n', lay);
        
        all_subject_pc1s = nan(length(subjects), num_voxels, 'single');
        
        for subj_idx = 1:length(subjects)
            subject = subjects{subj_idx};
            weight_file = fullfile(dataroot, subject, ['W_', feature_type, num2str(lay), '_PC1_norm.mat']);
            
            if exist(weight_file, 'file')
                W = load(weight_file).pc1_loading; % 1x59412
                all_subject_pc1s(subj_idx, :) =  W;
            else
                warning('  Missing weight file for subject %s, layer %d. Skipping.', subject, lay);
            end
        end
        
        % Average the PC1 loading vectors across all subjects for this layer
        group_beta_map = nanmean(all_subject_pc1s, 1);
        layer_wise_group_beta(lay+1, :) = group_beta_map;
    end
    
    % --- Save Group-Average Beta Map Results ---
    fprintf('\n  Saving final group-average beta map for %s...\n', feature_type);
    
    % Save as .mat file 
    save(fullfile(dataroot, ['group_beta_map_', feature_type, '.mat']), 'layer_wise_group_beta', '-v7.3');
    
    % Save as .dtseries.nii file
    cii = cii_template;
    cii.cdata = layer_wise_group_beta'; % Transpose to [Voxels x Layers]
    cii.diminfo{1, 1}.length = size(cii.cdata, 1);
    cii.diminfo{1, 2}.length = size(cii.cdata, 2);
    cii.diminfo{1, 1}.models(3:end) = [];

    beta_map_save_path = fullfile(dataroot, ['group_beta_map_', feature_type, '.dtseries.nii']);
    ciftisavereset(cii, beta_map_save_path, 'wb_command');
    
    fprintf('  Group beta map saved successfully.\n');
end

fprintf('\n--- All processing complete! ---\n');

