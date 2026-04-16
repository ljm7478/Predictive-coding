clc; clear;

% Paths
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/spm12'));
addpath(genpath('/local_raid1/01_software/spm12'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));

feature_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/Predictable_stimulus/E';
fmri_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding/';
base_save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/Predictable_stimulus/';

% layername = {'/E0';'/E1';'/E2';'/E3';};
layername = {'/E0';'/E1';'/E2';'/E3';'/E4';'/E5';'/E6';};

% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1'; 
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Exclude tasks
exclude_tasks = {'GarborPhasePred', 'GarborPhaseUnpred'};
   
% Stimulus mapping
stimulus_mapping = struct(...
    'circle', 'CirclePred', ...
    'garbor_frequency', 'GarborFreqPred', ...
    'garbor_orientation', 'GarborOrientationPred', ...
    'garbor_sc', 'GarborSCPred', ...
    'movie', 'MoviePred');


% Hyperparameters
lambda = [0.1:0.2:0.9];
nfold = 9;

%% Process Each Configuration
for config_idx = 1:size(configs, 1)
    train_dates = configs{config_idx, 1};
    test_date = configs{config_idx, 2};
    train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);

    % Dynamic run_numbers based on train_dates
    if ismember('241119_noT1', train_dates)
        run_numbers = struct('CirclePred', '01',  ...
            'GarborFreqPred', '05',  ...
            'GarborOrientationPred', '09',  ...
            'GarborSCPred', '07');
        
        % run_numbers = struct('MoviePred', '11');

        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    else
        run_numbers = struct('CirclePred', '01',  ...
            'GarborFreqPred', '05',...
            'GarborOrientationPred', '03',  ...
            'GarborSCPred', '07');

        % run_numbers = struct('MoviePred', '09');

        train_tasks = fieldnames(run_numbers);
    end

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    % Train voxelwise encoding model
    for lay = 1:length(layername)
        if length(layername) == 7
            layertype = 'layer7/E/';
        else
            layertype = 'layer4/E/';
        end

        % Ensure `saveroot` exists
        save_path = fullfile(base_save_path, layertype);
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end

        disp(['Processing Layer: ', layername{lay}]);

        % Iterate over all stimuli
        for stim = fieldnames(stimulus_mapping)'
            stim_name = stim{1};
            task_name = stimulus_mapping.(stim_name)

            % Load features for the current stimulus
            feature_file = fullfile(save_path, ['PredNet_feature_maps_pcareduced_', stim_name, '_layer', num2str(lay), '.h5']);
            if ~exist(feature_file, 'file')
                warning(['Feature file missing for: ', stim_name]);
                continue;
            end
            Y_train = h5read(feature_file, [layername{lay}, '/data']);

            % Aggregate fMRI datasets
            fmri_file = fullfile(fmri_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_', task_name, '.mat']);
            if ~exist(fmri_file, 'file')
                warning(['FMRI file missing for: ', fmri_file]);
                continue;
            end
            load(fmri_file, 'fmri_train');
            fmri_train_avg = mean(fmri_train, 3);
                                                                                                                                                                           
            % Train encoding model
            [W, Rmat, Lambda] = voxelwise_encoding(Y_train, fmri_train_avg', lambda, nfold);

            % Save weights
            W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_', stim_name, '_layer', num2str(lay), '.mat']);
            save(W_file, 'W', '-v7.3');
        end
    end

    % Test encoding model
    for stim = fieldnames(stimulus_mapping)'
        stim_name = stim{1};
        task_name = stimulus_mapping.(stim_name);
        for lay = 1:length(layername)
            if length(layername) == 7
                layertype = 'layer7/E/';
            else
                layertype = 'layer4/E/';
            end

            % Ensure `saveroot` exists
            save_path = fullfile(base_save_path, layertype);
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end

            feature_file = fullfile(save_path, ['PredNet_feature_maps_pcareduced_', stim_name, '_layer', num2str(lay), '.h5']);
            fmri_file = fullfile(fmri_path, ['testing_fmri_', test_date(1:6), '_', task_name, '.mat']);
            if ~exist(feature_file, 'file') || ~exist(fmri_file, 'file')
                warning(['Missing test data for: ', stim_name]);
                continue;
            end
            Y_test = h5read(feature_file, [layername{lay}, '/data']);
            load(fmri_file, 'fmri_test');
            fmri_test_avg = mean(fmri_test, 3);

            W_file = fullfile(save_path, ['W_lambda_', strjoin(train_dates_clean, '_'), '_', stim_name, '_layer', num2str(lay), '.mat']);
            load(W_file, 'W');

            pred_fmri = Y_test * W;

            fmri_test_avg = fmri_test_avg';
            % Calculate correlations
            correlation_scores = arrayfun(@(v) corr(pred_fmri(:, v), fmri_test_avg(:, v)), 1:size(pred_fmri, 2));

            all_corr{lay} = correlation_scores;

            % Save correlations
            corr_file = fullfile(save_path, ['corr_lambda_', test_date(1:6), '_', task_name, '_layer', num2str(lay), '.mat']);
            save(corr_file, 'correlation_scores', '-v7.3');
        end
        % Save results back to CIFTI
        fmripath ='/combinelab2/03_user/jungmin/02_data/02_jmlee_fmri/02_fmri/Preprocessed/241119_noT1/output/ciftify/sub-01/MNINonLinear/Results/task-CirclePred_run-01/task-CirclePred_run-01_Atlas_s3.dtseries.nii';
        cii = ciftiopen(fmripath, 'wb_command');
        concat_corr = cat(1, all_corr{:});
        cii.cdata = concat_corr';
        save_file_name = fullfile(save_path, sprintf('%s_%s_lambda5_E.dtseries.nii', test_date(1:6),task_name ));
        ciftisavereset(cii, save_file_name, 'wb_command');
    end
end

disp('All configurations processed successfully.');

%% Average Correlation dtseries Across Dates for Each Task
disp('Averaging dtseries across dates for each task...');

% Path to stored dtseries files
dtseries_path = save_path;

% Iterate over all tasks
for stim = fieldnames(stimulus_mapping)'
    stim_name = stim{1};  % Task name
    task_name = stimulus_mapping.(stim_name);

    % Find all dtseries files for this task across different test dates
    dtseries_files = {};
    for config_idx = 1:size(configs, 1)
        test_date = configs{config_idx, 2};  % Test date
        dtseries_file = fullfile(dtseries_path, sprintf('%s_%s_lambda5_E.dtseries.nii', test_date(1:6), task_name));

        disp(dtseries_file);
        if exist(dtseries_file, 'file')
            dtseries_files{end+1} = dtseries_file; % Store file path
        else
            warning(['Missing dtseries file: ', dtseries_file]);
        end
    end

    % Check if we have files to process
    if isempty(dtseries_files)
        warning(['No dtseries files found for task: ', task_name]);
        continue;
    end

    % Load first dtseries file to initialize averaging
    disp(['Loading dtseries file: ', dtseries_files{1}]);
    cii = ciftiopen(dtseries_files{1}, 'wb_command');
    avg_cdata = cii.cdata;  % Initialize with first file's data

    % Sum correlation values from the remaining files
    for i = 2:length(dtseries_files)
        disp(['Loading dtseries file: ', dtseries_files{i}]);
        temp_nii = ciftiopen(dtseries_files{i}, 'wb_command');
        avg_cdata = avg_cdata + temp_nii.cdata;  % Accumulate correlation data
    end

    % Compute mean correlation across dates
    avg_cdata = avg_cdata / length(dtseries_files);

    % Store the averaged correlations back into CIFTI
    cii.cdata = avg_cdata;
    save_file_name = fullfile(dtseries_path, sprintf('%s_lambda5_E_avg.dtseries.nii', task_name));

    disp(['Saving averaged dtseries file: ', save_file_name]);
    ciftisavereset(cii, save_file_name, 'wb_command');
end

disp('Averaged dtseries files created successfully.');

