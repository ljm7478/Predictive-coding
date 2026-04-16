clc; clear;

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

% Venus
% addpath(genpath('/local_raid1/202511_SW/HCPpipelines/global/matlab/cifti-matlab'));
% addpath(genpath('/combinelab/01_software/toolboxes/spm12'));
% addpath(genpath('/local_raid1/202511_SW/spm12'));
% addpath(genpath('/combinelab/01_software/toolboxes/cifti-matlab'));

%% --- Configuration ---
% Define the feature type to analyze ('Ahat' or 'E')
FEATURE_TYPE = 'Ahat'; % CHANGE THIS TO 'E' to analyze error features

% --- Paths ---
FEATURE_DATA_ROOT = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4';
FMRI_DATA_ROOT = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/';
SAVE_ROOT = fullfile('/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/kitti_pretrain_result/LORO_PCAcap/layer4/', FEATURE_TYPE);
CIFTI_TEMPLATE_FILE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii';

% --- Layer Names (automatically set based on FEATURE_TYPE) ---
if strcmp(FEATURE_TYPE, 'Ahat')
    layer_names = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'};
elseif strcmp(FEATURE_TYPE, 'E')
    layer_names = {'/E0'; '/E1'; '/E2'; '/E3'};
else
    error('Unknown FEATURE_TYPE: %s. Use ''Ahat'' or ''E''.', FEATURE_TYPE);
end
num_layers = length(layer_names);

% Use all runs for leave-one-run-out CV
all_runs = 0:7;  % All 8 runs
n_runs = length(all_runs);

% PCA settings
% CRITICAL: MAX_PCA_COMPONENTS prevents overfitting when features >> training samples
% Rule of thumb: n_components < n_fmri_TRs / 3
% Forrest Gump LORO: 7 runs * ~450 TRs = ~3150 training TRs
% MAX_PCA_COMPONENTS = 500 < 3150/3 = 1050 -> safe from overfitting
% Reference: Wen et al. 2018 (Cerebral Cortex), Naselaris et al. 2011 (NeuroImage)
PCA_VARIANCE_THRESHOLD = 0.90;  % 90% variance target
MAX_PCA_COMPONENTS = 500;       % Hard cap (must be < training TRs / 3)

STIMULUS_FRAMERATE = 25; % Hz
FMRI_TR = 2.0;
DOWNSAMPLE_FACTOR = STIMULUS_FRAMERATE * FMRI_TR;
hrf_params = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/STIMULUS_FRAMERATE, hrf_params);
hrf = hrf(:);

lambdas = 0.1:0.2:0.9;
nfold = 3;
subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

%% --- 1. Compute Mean, Std, and PCA from ALL Data ---
fprintf('\n========================================\n');
fprintf('STEP 1: Computing Mean, Std, and PCA from ALL runs\n');
fprintf('========================================\n');

if ~exist(SAVE_ROOT, 'dir'), mkdir(SAVE_ROOT); end

% Clean up existing files to allow fresh write
mean_file = fullfile(SAVE_ROOT, 'feature_mean.h5');
std_file = fullfile(SAVE_ROOT, 'feature_std.h5');
if exist(mean_file, 'file')
    fprintf('Deleting existing feature_mean.h5...\n');
    delete(mean_file);
end
if exist(std_file, 'file')
    fprintf('Deleting existing feature_std.h5...\n');
    delete(std_file);
end

for lay = 1:num_layers
    fprintf('\n--- Processing Layer %d/%d: %s ---\n', lay, num_layers, layer_names{lay});

    fprintf('Loading and reshaping ALL run data...\n'); 
    features_cell_2d = cell(1, n_runs);

    % Load ALL runs
    for i = 1:n_runs
        run_idx = all_runs(i);
        filepath = fullfile(FEATURE_DATA_ROOT, ['seg', num2str(run_idx), '_', FEATURE_TYPE, '.h5']);
        if exist(filepath, 'file')
            features_4d = h5read(filepath, layer_names{lay}); % (C, H, W, T)
            dims = size(features_4d);
            features_cell_2d{i} = single(reshape(features_4d, prod(dims(1:end-1)), dims(end)));
            fprintf('  Loaded run %d: %d features x %d timepoints\n', run_idx+1, size(features_cell_2d{i}, 1), size(features_cell_2d{i}, 2));
            clear features_4d;
        else
            error('Missing file: %s', filepath);
        end
    end

    % Concatenate ALL data
    all_features_2d = single(cat(2, features_cell_2d{:}));
    clear features_cell_2d;
    fprintf('Total: %d features x %d timepoints\n', size(all_features_2d, 1), size(all_features_2d, 2));

    % Compute mean and std from ALL data
    fprintf('Computing mean and std from ALL data...\n');
    lay_feat_mean_vec = mean(all_features_2d, 2);
    lay_feat_std_vec = std(all_features_2d, 0, 2);
    lay_feat_std_vec(lay_feat_std_vec == 0) = 1;  % Avoid division by zero

    % Reshape for saving (match original 4D structure minus time dimension)
    template_dims = size(h5read(fullfile(FEATURE_DATA_ROOT, ['seg0_', FEATURE_TYPE, '.h5']), layer_names{lay}));
    lay_feat_mean = reshape(lay_feat_mean_vec, template_dims(1:end-1));
    lay_feat_std = reshape(lay_feat_std_vec, template_dims(1:end-1));
  
    % Save mean and std
    h5_write_safe(mean_file, layer_names{lay}, single(lay_feat_mean));
    h5_write_safe(std_file, layer_names{lay}, single(lay_feat_std));

    % Compute PCA from ALL data with variance threshold AND max cap
    fprintf('Computing PCA components from ALL data...\n');
    standardized_features = bsxfun(@minus, all_features_2d, lay_feat_mean_vec);
    standardized_features = bsxfun(@rdivide, standardized_features, lay_feat_std_vec);
    standardized_features(isnan(standardized_features)) = 0;
    clear lay_feat_mean_vec lay_feat_std_vec all_features_2d; 

    [B, s] = compute_pca(standardized_features, PCA_VARIANCE_THRESHOLD, MAX_PCA_COMPONENTS);
    
    fprintf('Final PCA: %d components\n', size(B, 2));

    % Save PCA components
    save(fullfile(SAVE_ROOT, ['pca_components_layer', num2str(lay), '.mat']), 'B', 's', '-v7.3');
    
    clear standardized_features;
end
fprintf('\n--- Mean, Std, and PCA computation complete (using ALL runs). ---\n');

%% --- 2. Generate Regressors for All Runs ---
fprintf('\n========================================\n');
fprintf('STEP 2: Generating Regressors for All Runs\n');
fprintf('========================================\n');

% Clean up existing regressor files
regressor_save_path = fullfile(SAVE_ROOT, 'regressors');
if exist(regressor_save_path, 'dir')
    fprintf('Cleaning existing regressor directory...\n');
    delete(fullfile(regressor_save_path, '*.h5'));
else
    mkdir(regressor_save_path);
end

for lay = 1:num_layers
    fprintf('\nGenerating regressors for layer %s...\n', layer_names{lay});

    % Load pre-computed statistics
    lay_feat_mean = h5read(mean_file, layer_names{lay});
    lay_feat_std = h5read(std_file, layer_names{lay});
    load(fullfile(SAVE_ROOT, ['pca_components_layer', num2str(lay), '.mat']), 'B');

    for run_idx = all_runs
        filepath = fullfile(FEATURE_DATA_ROOT, ['seg', num2str(run_idx), '_', FEATURE_TYPE, '.h5']);
        if ~exist(filepath, 'file')
            warning('Missing file: %s', filepath);
            continue;
        end

        features = h5read(filepath, layer_names{lay});
        regressors = generate_regressors(features, lay_feat_mean, lay_feat_std, B, hrf, STIMULUS_FRAMERATE, DOWNSAMPLE_FACTOR);

        h5_write_safe(fullfile(regressor_save_path, ['run', num2str(run_idx+1), '.h5']), layer_names{lay}, single(regressors));
        fprintf('  Saved regressor for run %d: %d TRs x %d features\n', run_idx+1, size(regressors, 1), size(regressors, 2));
    end
end
fprintf('\n--- Regressor generation complete. ---\n');

%% --- 3. Summary ---
fprintf('\n========================================\n');
fprintf('PREPROCESSING COMPLETE\n');
fprintf('========================================\n');
fprintf('Feature type: %s\n', FEATURE_TYPE);
fprintf('Layers processed: %d\n', num_layers);
fprintf('Method: Leave-One-Run-Out CV preparation\n');
fprintf('Total runs: %d\n', n_runs);
fprintf('Mean/Std/PCA computed from: ALL %d runs\n', n_runs);
fprintf('PCA settings:\n');
fprintf('  - Variance threshold: %.0f%%\n', PCA_VARIANCE_THRESHOLD * 100);
fprintf('  - Max components cap: %d\n', MAX_PCA_COMPONENTS);
fprintf('Regressors generated for: ALL %d runs\n', n_runs);
fprintf('Save location: %s\n', SAVE_ROOT);
fprintf('\nNext step: Run encoding with leave-one-run-out CV\n');
fprintf('  - Each fold: Train on %d runs, Test on 1 run\n', n_runs-1);
fprintf('  - Total folds: %d\n', n_runs);
fprintf('========================================\n');


%% ========== HELPER FUNCTIONS ==========

function [B, s] = compute_pca(data, variance_threshold, max_components)
    % Compute PCA with variance threshold AND max component cap
    %
    % INPUTS:
    %   data: (n_features x n_timepoints) matrix, already standardized
    %   variance_threshold: target cumulative variance (e.g., 0.90 for 90%)
    %   max_components: hard cap on number of components
    %
    % OUTPUTS:
    %   B: (n_features x n_components) projection matrix
    %   s: eigenvalues
    %
    % CRITICAL: max_components prevents overfitting when components > fMRI samples
    %
    % Rule of thumb: n_components < n_fmri_TRs / 3
    %
    % References:
    %   - Wen et al. 2018 (Cerebral Cortex): "PCA... mitigated the potential 
    %     risk of overfitting with limited training data"
    %   - Naselaris et al. 2011 (NeuroImage): Encoding/decoding framework
    
    if nargin < 3
        max_components = 500;
    end
    
    fprintf('  PCA input: [%d x %d]\n', size(data, 1), size(data, 2));
    
    if size(data, 1) > size(data, 2)
        % More features than timepoints - use efficient covariance
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
    
    clear S R

    % Ensure non-negative eigenvalues
    s = max(s, 0);
    
    % Find number of components for variance threshold
    ratio = cumsum(s) / sum(s);
    Nc = find(ratio >= variance_threshold, 1, 'first');
    if isempty(Nc)
        Nc = length(s);
    end
    
    % Report before capping
    Nc_original = Nc;
    fprintf('  %.0f%% variance requires %d components\n', variance_threshold*100, Nc_original);
    
    % Apply max cap (CRITICAL for preventing overfitting)
    Nc = min(Nc, max_components);
    
    if Nc < Nc_original
        fprintf('  *** CAPPED to %d (max_components = %d) ***\n', Nc, max_components);
    end
    fprintf('  Actual variance retained: %.2f%%\n', ratio(Nc)*100);
    
    % Construct projection matrix B
    if size(data, 1) > size(data, 2)
        s_subset = max(s(1:Nc), eps);
        S_inv_sqrt = diag(1 ./ sqrt(s_subset));
        B = data * (U(:, 1:Nc) * S_inv_sqrt / sqrt(size(data, 1)));
        clear U S_inv_sqrt; 
    else
        B = U(:, 1:Nc);
        clear U;
    end
    
    fprintf('  B output: [%d x %d]\n', size(B, 1), size(B, 2));
end


function regressors = generate_regressors(features_4d, f_mean, f_std, B, hrf, srate, downsample_factor)
    % Generate regressors: standardize -> PCA project -> HRF convolve -> downsample
    %
    % INPUTS:
    %   features_4d: (C, H, W, T) raw feature tensor
    %   f_mean: (C, H, W) mean computed from ALL runs
    %   f_std: (C, H, W) std computed from ALL runs
    %   B: (n_features x n_components) PCA projection matrix
    %   hrf: hemodynamic response function
    %   srate: stimulus frame rate (Hz)
    %   downsample_factor: frames per fMRI TR
    %
    % OUTPUT:
    %   regressors: (n_TRs x n_components) regressor matrix
    
    % Step 1: Standardize features using pre-computed mean/std
    features_standardized = bsxfun(@minus, features_4d, f_mean);
    features_standardized = bsxfun(@rdivide, features_standardized, f_std);
    features_standardized(isnan(features_standardized)) = 0;
    
    % Reshape to 2D: (n_features x n_timepoints)
    dims = size(features_standardized);
    features_reshaped = reshape(features_standardized, prod(dims(1:end-1)), dims(end));
    
    % Step 2: Project to PCA space
    Y = features_reshaped' * B;  % (n_timepoints x n_components)

    clear features_reshaped features_standardized;
    
    % Step 3: Convolve with HRF
    ts = conv2(Y, hrf, 'full');
    
    % Adjust for HRF delay (peak at ~4-6 seconds)
    hrf_peak_delay_frames = 4 * srate;  % 4 seconds * frame rate
    ts = ts(hrf_peak_delay_frames + 1 : hrf_peak_delay_frames + size(Y, 1), :);
    
    % Step 4: Downsample to fMRI TR
    regressors = ts(round(1:downsample_factor:end), :);
end


function h5_write_safe(filepath, dataset_name, data)
    % Safe HDF5 write - creates directory and handles existing datasets
    %
    % INPUTS:
    %   filepath: path to .h5 file
    %   dataset_name: name of dataset (e.g., '/Ahat0')
    %   data: data to write
    
    [folder, ~, ~] = fileparts(filepath);
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
    % Check if file exists
    if ~exist(filepath, 'file')
        % File doesn't exist - create new
        h5create(filepath, dataset_name, size(data), 'Datatype', class(data));
        h5write(filepath, dataset_name, data);
    else
        % File exists - check if dataset exists
        try
            h5info(filepath, dataset_name);
            % Dataset exists - skip (files should be cleaned before running)
            warning('Dataset %s already exists in %s. Skipping.', dataset_name, filepath);
            return;
        catch
            % Dataset doesn't exist - add to existing file
            h5create(filepath, dataset_name, size(data), 'Datatype', class(data));
            h5write(filepath, dataset_name, data);
        end
    end
end