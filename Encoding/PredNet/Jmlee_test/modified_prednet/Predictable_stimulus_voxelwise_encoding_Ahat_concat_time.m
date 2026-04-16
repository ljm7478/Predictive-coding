clc; clear;

% Paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

fmri_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding_concat_time_noHRF/garbor';
save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Predictable_stimulus/layer7/Ahat/garbor/';
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3';};
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6';};

% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1'; 
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Hyperparameters
lambda = [0.1:0.2:0.9];
nfold = 3; %original:9

% Stimulus groups (excluding movie tasks)
stimulus_mapping = struct(...
    'circle', 'CirclePred', ...
    'garbor_frequency', 'GarborFreqPred', ...
    'garbor_orientation', 'GarborOrientationPred', ...
    'garbor_sc', 'GarborSCPred');

%% Process Each Configuration
clc;
for config_idx = 1:size(configs, 1)
    train_dates = configs{config_idx, 1};
    test_date = configs{config_idx, 2};
    train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    % Train voxelwise encoding model
    for lay = 1:length(layername)
        disp(['Processing Layer: ', layername{lay}]);

        % Initialize Y_train
        Y_train = [];

        % Iterate over stimuli
        for stim = fieldnames(stimulus_mapping)'
            stim_key = stim{1};  % Current stimulus
            task_name = stimulus_mapping.(stim_key);
            if contains(lower(task_name), 'pred')
                group_type = 'Pred';
            else
                group_type = 'Unpred';
            end
            % Construct feature and fMRI file paths
            feature_file = fullfile(save_path, ['PredNet_feature_maps_processed_', stim_key, '_layer', num2str(lay), '.h5']);
            fmri_file = fullfile(fmri_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_', group_type, '.mat']);

            % Debug: Print file paths
            disp(['Feature file path: ', feature_file]);
            disp(['fMRI file path: ', fmri_file]);

            if exist(feature_file, 'file') && exist(fmri_file, 'file')
                % Load features and fMRI data
                Y_data = h5read(feature_file, [layername{lay}, '/data']);  % Load feature data
                load(fmri_file, 'fmri_train_pred');  % Load fMRI data

                % Concatenate feature data along its temporal dimension
                if isempty(Y_train)
                    Y_train = Y_data;  % Initialize Y_train
                else
                    Y_train = [Y_train; Y_data];  % Concatenate along time dimension
                end

                % Average fMRI data across the repetition dimension
                fmri_train_avg = mean(fmri_train_pred, 3);  % Average across 3rd dimension
            else
                warning(['Missing data for stimulus: ', stim_key]);
            end
        end

        % At this point:
        % Y_train contains the concatenated feature data across stimuli
        % fmri_train_avg contains the averaged fMRI data

        % Train encoding model
        [W, Rmat, Lambda] = voxelwise_encoding(Y_train, fmri_train_avg', lambda, nfold);

        % Save weights
        W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_layer', num2str(lay), '.mat']);
        save(W_file, 'W', '-v7.3');
        disp(['Saved weights: ', W_file]);
    end

    % Test voxelwise encoding model
    for lay = 1:length(layername)
        disp(['Testing Layer: ', layername{lay}]);

        Y_test = [];
        fmri_test = [];

        % Process each stimulus in stimulus_mapping
        for stim_name = fieldnames(stimulus_mapping)'
            stim_key = stim_name{1};
            task_name = stimulus_mapping.(stim_key);
            if contains(lower(task_name), 'pred')
                group_type = 'Pred';
            else
                group_type = 'Unpred';
            end

            % Construct feature and fMRI file paths
            feature_file = fullfile(save_path, ['PredNet_feature_maps_processed_', stim_key, '_layer', num2str(lay), '.h5']);
            fmri_file = fullfile(fmri_path, ['testing_fmri_', test_date(1:6), '_', group_type, '.mat']);

            % Debug: Print file paths
            disp(['Feature file path: ', feature_file]);
            disp(['fMRI file path: ', fmri_file]);

            if exist(feature_file, 'file') && exist(fmri_file, 'file')
                % Load features and fMRI data
                Y_data = h5read(feature_file, [layername{lay}, '/data']);
                load(fmri_file, 'fmri_test_pred');

                % Concatenate data
                Y_test = [Y_test; Y_data];
                fmri_test = cat(3, fmri_test, fmri_test_pred);
            else
                warning(['Missing test data for stimulus: ', stim_key]);
            end
        end

        if isempty(fmri_test) || isempty(Y_test)
            warning('No test data available for this layer.');
            continue;
        end

        % Load weights
        W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_layer', num2str(lay), '.mat']);
        if ~exist(W_file, 'file')
            warning(['Weights file missing: ', W_file]);
            continue;
        end
        load(W_file, 'W');

        % Predict fMRI responses
        fmri_test_avg = mean(fmri_test, 3);
        pred_fmri = Y_test * W;

        % Calculate correlations
        fmri_test_avg = fmri_test_avg';
        correlation_scores = arrayfun(@(v) corr(pred_fmri(:, v), fmri_test_avg(:, v)), 1:size(pred_fmri, 2));

        % Save correlations
        corr_file = fullfile(save_path, ['corr_lambda_', test_date(1:6), '_', task_name(9:end),'_layer', num2str(lay), '.mat']);
        save(corr_file, 'correlation_scores', '-v7.3');
        disp(['Saved correlation scores: ', corr_file]);
    end
end

disp('All configurations processed successfully.');

%% cdata
clc;
% Paths for CIFTI and correlation files
corr_path ='/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Predictable_stimulus/layer7/Ahat/garbor/';
fmripath = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/02_fmri/Preprocessed/241119_noT1/output/ciftify/sub-01/MNINonLinear/Results/task-CirclePred_run-01/task-CirclePred_run-01_Atlas_s3.dtseries.nii';

% Initialize storage for concatenated correlations
all_corr_avg = cell(1, length(layername));

% Iterate over layers
for lay = 1:length(layername)
    disp(['Processing Layer: ', layername{lay}]);

    % Initialize to collect correlation scores from multiple dates
    corr_all_dates = [];

    % Iterate over test dates
    for config_idx = 1:size(configs, 1)
        test_date = configs{config_idx, 2};
        corr_file = fullfile(corr_path, ['corr_lambda_', test_date(1:6), '_Pred_layer', num2str(lay), '.mat']);

        if exist(corr_file, 'file')
            disp(['Loading correlation file: ', corr_file]);
            load(corr_file, 'correlation_scores');

            % Collect correlation scores for averaging across dates
            if isempty(corr_all_dates)
                corr_all_dates = correlation_scores;
            else
                corr_all_dates = cat(1, corr_all_dates, correlation_scores);
            end
        else
            warning(['Correlation file missing for date: ', test_date]);
        end
    end

    % Average correlation scores across dates
    if ~isempty(corr_all_dates)
        avg_corr = mean(corr_all_dates, 1);
        all_corr_avg{lay} = avg_corr;
    else
        warning(['No correlation data found for layer ', num2str(lay)]);
    end
end

% Concatenate averaged correlations across all layers
concat_corr = cat(1, all_corr_avg{:});

% Load CIFTI file for saving
cii = ciftiopen(fmripath, 'wb_command');

% Update CIFTI data with concatenated correlations
disp('Updating CIFTI structure with concatenated correlations...');
cii.cdata = concat_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);

% Save the updated CIFTI file
save_file_name = fullfile(corr_path, 'lambda5_Ahat_avg_corr.dtseries.nii');
disp(['Saving updated CIFTI file: ', save_file_name]);
ciftisavereset(cii, save_file_name, 'wb_command');

disp('CIFTI file with averaged correlations saved successfully.');


%% Movie task 

clc; clear;

% Paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Ahat/movie';
fmri_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding_concat_time_noHRF/movie';
save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Predictable_stimulus/layer7/Ahat/movie/';
% layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3';};
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'; '/Ahat4'; '/Ahat5'; '/Ahat6';};

% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1'; 
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Hyperparameters
lambda = [0.1:0.2:0.9];
nfold = 3;

% Stimulus groups (excluding movie tasks)
stimulus_mapping = struct('movie', 'MoviePred');

%% Process Each Configuration
clc;
for config_idx = 1:size(configs, 1)
    train_dates = configs{config_idx, 1};
    test_date = configs{config_idx, 2};
    train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    % Train voxelwise encoding model
    for lay = 1:length(layername)
        disp(['Processing Layer: ', layername{lay}]);

        % Initialize Y_train
        Y_train = [];

        % Iterate over stimuli
        for stim = fieldnames(stimulus_mapping)'
            stim_key = stim{1};  % Current stimulus
            task_name = stimulus_mapping.(stim_key);
            if contains(lower(task_name), 'pred')
                group_type = 'Pred';
            else
                group_type = 'Unpred';
            end
            % Construct feature and fMRI file paths
            feature_file = fullfile(save_path, ['PredNet_feature_maps_processed_', stim_key, '_layer', num2str(lay), '.h5']);
            fmri_file = fullfile(fmri_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_', group_type, '.mat']);

            % Debug: Print file paths
            disp(['Feature file path: ', feature_file]);
            disp(['fMRI file path: ', fmri_file]);

            if exist(feature_file, 'file') && exist(fmri_file, 'file')
                % Load features and fMRI data
                Y_data = h5read(feature_file, [layername{lay}, '/data']);  % Load feature data
                load(fmri_file, 'fmri_train_pred');  % Load fMRI data

                % Concatenate feature data along its temporal dimension
                if isempty(Y_train)
                    Y_train = Y_data;  % Initialize Y_train
                else
                    Y_train = [Y_train; Y_data];  % Concatenate along time dimension
                end

                % Average fMRI data across the repetition dimension
                fmri_train_avg = mean(fmri_train_pred, 3);  % Average across 3rd dimension
            else
                warning(['Missing data for stimulus: ', stim_key]);
            end
        end

        % At this point:
        % Y_train contains the concatenated feature data across stimuli
        % fmri_train_avg contains the averaged fMRI data

        % Train encoding model
        [W, Rmat, Lambda] = voxelwise_encoding(Y_train, fmri_train_avg', lambda, nfold);

        % Save weights
        W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_layer', num2str(lay), '.mat']);
        save(W_file, 'W', '-v7.3');
        disp(['Saved weights: ', W_file]);
    end

    % Test voxelwise encoding model
    for lay = 1:length(layername)
        disp(['Testing Layer: ', layername{lay}]);

        Y_test = [];
        fmri_test = [];

        % Process each stimulus in stimulus_mapping
        for stim_name = fieldnames(stimulus_mapping)'
            stim_key = stim_name{1};
            task_name = stimulus_mapping.(stim_key);
            if contains(lower(task_name), 'pred')
                group_type = 'Pred';
            else
                group_type = 'Unpred';
            end
 
            % Construct feature and fMRI file paths
            feature_file = fullfile(save_path, ['PredNet_feature_maps_processed_', stim_key, '_layer', num2str(lay), '.h5']);
            fmri_file = fullfile(fmri_path, ['testing_fmri_', test_date(1:6), '_', group_type, '.mat']);

            % Debug: Print file paths
            disp(['Feature file path: ', feature_file]);
            disp(['fMRI file path: ', fmri_file]);

            if exist(feature_file, 'file') && exist(fmri_file, 'file')
                % Load features and fMRI data
                Y_data = h5read(feature_file, [layername{lay}, '/data']);
                load(fmri_file, 'fmri_test_pred');

                % Concatenate data
                Y_test = [Y_test; Y_data];
                fmri_test = cat(3, fmri_test, fmri_test_pred);
            else
                warning(['Missing test data for stimulus: ', stim_key]);
            end
        end

        if isempty(fmri_test) || isempty(Y_test)
            warning('No test data available for this layer.');
            continue;
        end

        % Load weights
        W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_layer', num2str(lay), '.mat']);
        if ~exist(W_file, 'file')
            warning(['Weights file missing: ', W_file]);
            continue;
        end
        load(W_file, 'W');

        % Predict fMRI responses
        fmri_test_avg = mean(fmri_test, 3);
        pred_fmri = Y_test * W;

        % Calculate correlations
        fmri_test_avg = fmri_test_avg';
        correlation_scores = arrayfun(@(v) corr(pred_fmri(:, v), fmri_test_avg(:, v)), 1:size(pred_fmri, 2));

        % Save correlations
        corr_file = fullfile(save_path, ['corr_lambda_', test_date(1:6), '_', task_name(6:end),'_layer', num2str(lay), '.mat']);
        save(corr_file, 'correlation_scores', '-v7.3');
        disp(['Saved correlation scores: ', corr_file]);
    end
end

disp('All configurations processed successfully.');

%% cdata
clc;
% Paths for CIFTI and correlation files
corr_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Predictable_stimulus/layer7/Ahat/movie/';
fmripath = '/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/02_fmri/Preprocessed/241119_noT1/output/ciftify/sub-01/MNINonLinear/Results/task-CirclePred_run-01/task-CirclePred_run-01_Atlas_s3.dtseries.nii';

% Initialize storage for concatenated correlations
all_corr_avg = cell(1, length(layername));

% Iterate over layers
for lay = 1:length(layername)
    disp(['Processing Layer: ', layername{lay}]);

    % Initialize to collect correlation scores from multiple dates
    corr_all_dates = [];

    % Iterate over test dates
    for config_idx = 1:size(configs, 1)
        test_date = configs{config_idx, 2};
        corr_file = fullfile(corr_path, ['corr_lambda_', test_date(1:6), '_Pred_layer', num2str(lay), '.mat']);

        if exist(corr_file, 'file')
            disp(['Loading correlation file: ', corr_file]);
            load(corr_file, 'correlation_scores');

            % Collect correlation scores for averaging across dates
            if isempty(corr_all_dates)
                corr_all_dates = correlation_scores;
            else
                corr_all_dates = cat(1, corr_all_dates, correlation_scores);
            end
        else
            warning(['Correlation file missing for date: ', test_date]);
        end
    end

    % Average correlation scores across dates
    if ~isempty(corr_all_dates)
        avg_corr = mean(corr_all_dates, 1);
        all_corr_avg{lay} = avg_corr;
    else
        warning(['No correlation data found for layer ', num2str(lay)]);
    end
end

% Concatenate averaged correlations across all layers
concat_corr = cat(1, all_corr_avg{:});

% Load CIFTI file for saving
cii = ciftiopen(fmripath, 'wb_command');

% Update CIFTI data with concatenated correlations
disp('Updating CIFTI structure with concatenated correlations...');
cii.cdata = concat_corr';
cii.diminfo{1, 1}.length = size(cii.cdata, 1);
cii.diminfo{1, 2}.length = size(cii.cdata, 2);

% Save the updated CIFTI file
save_file_name = fullfile(corr_path, 'lambda5_Ahat_avg_corr.dtseries.nii');
disp(['Saving updated CIFTI file: ', save_file_name]);
ciftisavereset(cii, save_file_name, 'wb_command');

disp('CIFTI file with averaged correlations saved successfully.');