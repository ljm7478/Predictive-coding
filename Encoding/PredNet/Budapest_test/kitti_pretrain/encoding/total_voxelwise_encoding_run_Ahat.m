%% Voxel-wise Encoding Pipeline for Budapest Dataset
%
% This script performs voxel-wise encoding analysis:
% 1. Feature Standardization: Computes mean and std of training features
% 2. Dimensionality Reduction: PCA with variance threshold + max component cap
% 3. Regressor Generation: PCA projection + HRF convolution + downsampling
% 4. Encoding Model Training & Evaluation
%
% IMPORTANT: MAX_PCA_COMPONENTS prevents overfitting when features > fMRI samples

clc; clear;

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

%% --- Configuration ---

FEATURE_TYPE = 'Ahat';  % 'E' for predictions, 'E' for errors

% --- Paths ---
FEATURE_DATA_ROOT = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/08_Budapest/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4';
FMRI_DATA_ROOT = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/fmri_new/';
SAVE_ROOT = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/', FEATURE_TYPE);
CIFTI_TEMPLATE_FILE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';

% --- Layer Configuration ---
layer_names = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
num_layers = length(layer_names);

% --- Data Split (Budapest: 5 runs) ---
% Run TRs: seg1=598, seg2=498, seg3=535, seg4=618, seg5=803
train_runs = 1:3;  % seg1, seg2, seg3 -> 598 + 498 + 535 = 1631 TRs
test_runs = 4:5;   % seg4, seg5 -> 618 + 803 = 1421 TRs

% --- PCA Parameters ---
% CRITICAL FOR PREVENTING OVERFITTING:
%
% Budapest fMRI training samples: 1631 TRs
% If PCA components > 1631 -> underdetermined regression -> overfitting -> r ?? 0
%
% Rule of thumb: MAX_PCA_COMPONENTS <= fMRI_TRs / 3 = 1631 / 3 ?? 544

PCA_VARIANCE_THRESHOLD = 0.90;  % Target variance to explain
MAX_PCA_COMPONENTS = 500;       % Hard cap (must be < training TRs = 1631)

% --- Budapest-specific Parameters ---
STIMULUS_FRAMERATE = 30;  % Hz (Budapest frame rate)
FMRI_TR = 1.0;            % seconds (Budapest TR)
DOWNSAMPLE_FACTOR = STIMULUS_FRAMERATE * FMRI_TR;  % 30 frames per TR

hrf_params = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/STIMULUS_FRAMERATE, hrf_params);
hrf = hrf(:);

% --- Encoding Parameters ---
lambdas = 0.1:0.2:0.9;
nfold = 3;

subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', ...
            'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', ...
            'sub-18', 'sub-19', 'sub-20'};


%% --- 1. Compute Mean, Std, and PCA from Training Data ---

if ~exist(SAVE_ROOT, 'dir'), mkdir(SAVE_ROOT); end

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('[Step 1] Computing Mean, Std, and PCA from Training Data\n');
fprintf('%s\n', repmat('=', 1, 70));

for lay = 1:num_layers
    fprintf('\n--- Layer %d/%d (%s) ---\n', lay, num_layers, layer_names{lay});

    % Load and reshape training data
    fprintf('Loading training data...\n');
    features_cell_2d = cell(1, length(train_runs));

    for i = 1:length(train_runs)
        run_idx = train_runs(i);
        filepath = fullfile(FEATURE_DATA_ROOT, ['seg', num2str(run_idx), '_', FEATURE_TYPE, '.h5']);
        
        if exist(filepath, 'file')
            % Load 4D features: (C, H, W, T)
            features_4d = h5read(filepath, layer_names{lay});
            dims = size(features_4d);
            
            % Reshape to 2D: (n_features, n_timepoints)
            features_cell_2d{i} = reshape(features_4d, prod(dims(1:end-1)), dims(end));
            fprintf('  Run %d (seg%d): [%d features x %d timepoints]\n', ...
                i, run_idx, size(features_cell_2d{i}, 1), size(features_cell_2d{i}, 2));
        else
            error('Missing training file: %s', filepath);
        end
    end

    % Concatenate all training runs
    all_train_features_2d = cat(2, features_cell_2d{:});
    clear features_cell_2d;
    
    fprintf('Concatenated: [%d features x %d timepoints]\n', ...
        size(all_train_features_2d, 1), size(all_train_features_2d, 2));

    % Compute mean and std
    fprintf('Computing mean and std...\n');
    lay_feat_mean_vec = mean(all_train_features_2d, 2);
    lay_feat_std_vec = std(all_train_features_2d, 0, 2);
    lay_feat_std_vec(lay_feat_std_vec == 0) = 1;

    % Reshape for saving
    template_dims = size(h5read(fullfile(FEATURE_DATA_ROOT, ['seg1_', FEATURE_TYPE, '.h5']), layer_names{lay}));
    lay_feat_mean = reshape(lay_feat_mean_vec, template_dims(1:end-1));
    lay_feat_std = reshape(lay_feat_std_vec, template_dims(1:end-1));

    % Save mean and std
    h5_write_safe(fullfile(SAVE_ROOT, 'feature_mean.h5'), layer_names{lay}, single(lay_feat_mean));
    h5_write_safe(fullfile(SAVE_ROOT, 'feature_std.h5'), layer_names{lay}, single(lay_feat_std));

    % Standardize features
    fprintf('Computing PCA...\n');
    standardized_features = bsxfun(@minus, all_train_features_2d, lay_feat_mean_vec);
    standardized_features = bsxfun(@rdivide, standardized_features, lay_feat_std_vec);
    standardized_features(isnan(standardized_features)) = 0;

    % Compute PCA with max cap
    [B, s] = compute_pca(standardized_features, PCA_VARIANCE_THRESHOLD, MAX_PCA_COMPONENTS);
    
    fprintf('B output: [%d x %d]\n', size(B, 1), size(B, 2));

    % Save PCA components
    save(fullfile(SAVE_ROOT, ['pca_components_layer', num2str(lay), '.mat']), 'B', 's', '-v7.3');
    
    clear all_train_features_2d standardized_features;
end

fprintf('\n[Step 1] Complete.\n');


%% --- 2. Generate Regressors for All Runs ---

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('[Step 2] Generating Regressors for All Runs\n');
fprintf('%s\n', repmat('=', 1, 70));

all_runs = [train_runs, test_runs];
regressor_save_path = fullfile(SAVE_ROOT, 'regressors');
if ~exist(regressor_save_path, 'dir'), mkdir(regressor_save_path); end

for lay = 1:num_layers
    fprintf('\nLayer %s:\n', layer_names{lay});

    % Load pre-computed stats
    lay_feat_mean = h5read(fullfile(SAVE_ROOT, 'feature_mean.h5'), layer_names{lay});
    lay_feat_std = h5read(fullfile(SAVE_ROOT, 'feature_std.h5'), layer_names{lay});
    load(fullfile(SAVE_ROOT, ['pca_components_layer', num2str(lay), '.mat']), 'B');

    for run_idx = all_runs
        filepath = fullfile(FEATURE_DATA_ROOT, ['seg', num2str(run_idx), '_', FEATURE_TYPE, '.h5']);
        if ~exist(filepath, 'file')
            warning('Missing file: seg%d', run_idx);
            continue;
        end

        features = h5read(filepath, layer_names{lay});
        regressors = generate_regressors(features, lay_feat_mean, lay_feat_std, B, hrf, STIMULUS_FRAMERATE, DOWNSAMPLE_FACTOR);
        
        fprintf('  Run %d (seg%d): regressors [%d TRs x %d features]\n', ...
            run_idx, run_idx, size(regressors, 1), size(regressors, 2));

        h5_write_safe(fullfile(regressor_save_path, ['run', num2str(run_idx), '.h5']), ...
            layer_names{lay}, single(regressors));
    end
end

fprintf('\n[Step 2] Complete.\n');


%% --- 3. Run Voxel-wise Encoding for Each Subject ---
% cii_template = ciftiopen(CIFTI_TEMPLATE_FILE, 'wb_command');
% 
% for subj_idx = 2:length(subjects)
%     subject = subjects{subj_idx};
%     fprintf('\n--- Starting Encoding for Subject: %s ---\n', subject);
% 
%     subject_save_dir = fullfile(SAVE_ROOT, subject);
%     if ~exist(subject_save_dir, 'dir'), mkdir(subject_save_dir); end
% 
%    % --- Load per-run fMRI data and construct training set ---
%     fmri_per_run_path = fullfile(FMRI_DATA_ROOT, subject, [subject, '_per_run_fmri.mat']);
%     if ~exist(fmri_per_run_path, 'file'), warning('Missing per-run fMRI data for %s. Skipping.', subject); continue; end
%     fmri_data_struct = load(fmri_per_run_path).fmri_data_per_run;
% 
%     % Concatenate only the training runs for fMRI data
%     fmri_train_cell = cell(1, length(train_runs));
%     for i = 1:length(train_runs)
%         run_name = ['run', num2str(train_runs(i)+1)];
%         if isfield(fmri_data_struct, run_name)
%             fmri_train_cell{i} = fmri_data_struct.(run_name);
%         else
%             warning('Missing fMRI run %d for subject %s.', train_runs(i), subject);
%         end
%     end
%     fmri_train_data = cat(2, fmri_train_cell{:}); % This is now correctly (Voxels x TrainingTime)
% 
%     all_subject_corrs = cell(num_layers, 1);
% 
%     for lay = 1:num_layers
%         fprintf('  Encoding for layer %s...\n', layer_names{lay});
% 
%         train_regressors_cell = cell(1, length(train_runs));
%         for i = 1:length(train_runs)
%             run_idx = train_runs(i);
%             regressor_path = fullfile(SAVE_ROOT, 'regressors', ['run', num2str(run_idx+1), '.h5']);
%             train_regressors_cell{i} = h5read(regressor_path, layer_names{lay});
%         end
%         Y_train = cat(1, train_regressors_cell{:});
% 
%         [W, ~, ~] = voxelwise_encoding_new(Y_train, fmri_train_data', lambdas, nfold);
% 
%         % Save the W matrix for beta map analysis
%         save(fullfile(subject_save_dir, ['W_layer', num2str(lay), '.mat']), 'W', '-v7.3');
% 
%         test_corrs_per_run = [];
%         for i = 1:length(test_runs)
%             run_idx = test_runs(i);
%             run_name = ['run', num2str(run_idx+1)];
% 
%             regressor_path = fullfile(SAVE_ROOT, 'regressors', [run_name, '.h5']);
%             Y_test = h5read(regressor_path, layer_names{lay});
%             fmri_test = fmri_data_struct.(run_name);
% 
%             pred_fmri = Y_test * W;
% 
%             min_len = min(size(pred_fmri, 1), size(fmri_test, 2));
%             pred_fmri = pred_fmri(1:min_len, :);
%             fmri_test = fmri_test(:, 1:min_len)';
% 
%             corrs = diag(corr(pred_fmri, fmri_test));
%             corrs(isnan(corrs)) = 0;
%             test_corrs_per_run = [test_corrs_per_run, corrs];
%         end
% 
%         avg_corrs = mean(test_corrs_per_run, 2);
%         all_subject_corrs{lay} = avg_corrs';
%     end
% 
%     final_corr_matrix = cat(1, all_subject_corrs{:});
%     save(fullfile(subject_save_dir, 'average_correlations.mat'), 'final_corr_matrix', '-v7.3');
% 
%     cii = cii_template;
%     cii.cdata = final_corr_matrix';
%     cii.diminfo{1, 1}.length = size(cii.cdata, 1);
%     cii.diminfo{1, 2}.length = size(cii.cdata, 2);
%     cii.diminfo{1, 1}.models(3:end) = [];
%     ciftisavereset(cii, fullfile(subject_save_dir, 'average_correlations.dtseries.nii'), 'wb_command');
% 
%     fprintf('  Finished encoding for subject %s.\n', subject);
% end
% 
% fprintf('\n--- All subjects processed! ---\n');

%% --- Helper Functions ---

function h5_write_safe(filepath, dataset, data)
    if ~exist(fileparts(filepath), 'dir'), mkdir(fileparts(filepath)); end
    if exist(filepath, 'file')
        info = h5info(filepath);
        if any(strcmp({info.Datasets.Name}, dataset(2:end)))
             warning('Dataset %s already exists in %s. It will be overwritten.', dataset(2:end), filepath);
        end
    end
    h5create(filepath, dataset, size(data), 'Datatype', 'single');
    h5write(filepath, dataset, data);
end

function [B, s] = compute_pca(data, variance_threshold, max_components)
    % Compute PCA with variance threshold AND max component cap
    %
    % Inputs:
    %   data: (n_features, n_timepoints)
    %   variance_threshold: target variance to explain (e.g., 0.90)
    %   max_components: hard cap on components (prevents overfitting)
    
    if nargin < 3
        max_components = 500;  % Default cap
    end
    
    fprintf('  PCA input: [%d x %d]\n', size(data, 1), size(data, 2));
    
    if size(data, 1) > size(data, 2)
        % More features than timepoints
        fprintf('  IF branch: features (%d) > timepoints (%d)\n', size(data,1), size(data,2));
        R = data' * data / size(data, 1);
        [U, S, ~] = svd(R, 'econ');
        s = diag(S);
    else
        % More timepoints than features
        fprintf('  ELSE branch: timepoints (%d) >= features (%d)\n', size(data,2), size(data,1));
        R = data * data' / size(data, 2);
        [U, S, ~] = svd(R, 'econ');
        s = diag(S);
    end
    
    % Ensure non-negative eigenvalues
    s = max(s, 0);
    
    % Find components for variance threshold
    ratio = cumsum(s) / sum(s);
    Nc = find(ratio >= variance_threshold, 1, 'first');
    if isempty(Nc)
        Nc = length(s);
    end
    
    % Apply max cap (CRITICAL for preventing overfitting)
    Nc_original = Nc;
    Nc = min(Nc, max_components);
    
    fprintf('  %.0f%% variance requires %d components\n', variance_threshold*100, Nc_original);
    fprintf('  Capped to %d (max_components = %d)\n', Nc, max_components);
    fprintf('  Actual variance: %.2f%%\n', ratio(Nc)*100);
    
    % Construct projection matrix B
    if size(data, 1) > size(data, 2)
        s_subset = max(s(1:Nc), eps);
        S_inv_sqrt = diag(1 ./ sqrt(s_subset));
        B = data * (U(:, 1:Nc) * S_inv_sqrt / sqrt(size(data, 1)));
    else
        B = U(:, 1:Nc);
    end
end

function regressors = generate_regressors(features_4d, f_mean, f_std, B, hrf, srate, downsample_factor)
    features_standardized = bsxfun(@minus, features_4d, f_mean);
    features_standardized = bsxfun(@rdivide, features_standardized, f_std);
    features_standardized(isnan(features_standardized)) = 0;
    
    dims = size(features_standardized);
    features_reshaped = reshape(features_standardized, prod(dims(1:end-1)), dims(end));
    Y = features_reshaped' * B;
    
    ts = conv2(Y, hrf, 'full');
    
    hrf_peak_delay_frames = 4 * srate;
    ts = ts(hrf_peak_delay_frames + 1 : hrf_peak_delay_frames + size(Y, 1), :);
    
    regressors = ts(round(1:downsample_factor:end), :);
end

