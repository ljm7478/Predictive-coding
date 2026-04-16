clc; clear;
%% Garbor task
%% Parameters
Nv = 91282; % Number of voxels

save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding_concat_time_noHRF/garbor/';
data_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/noHRF/';

% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1';
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Default exclude tasks
exclude_tasks = {'GarborPhasePred', 'GarborPhaseUnpred', 'MoviePred', 'MovieUnpred'}; % Exclude movie and specific garbor tasks

%% Process fMRI Data for Each Configuration
for config_idx = 1:size(configs, 1)
    train_dates = configs{config_idx, 1};
    test_date = configs{config_idx, 2};

    % Dynamically assign run_numbers based on scanning_dates
    if ismember('241119_noT1', train_dates)
        run_numbers = struct(...
            'CirclePred', '01', 'CircleUnpred', '02', ...
            'GarborFreqPred', '05', 'GarborFreqUnpred', '06', ...
            'GarborOrientationPred', '09', 'GarborOrientationUnpred', '10', ...
            'GarborSCPred', '07', 'GarborSCUnpred', '08');
        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    else
        run_numbers = struct(...
            'CirclePred', '01', 'CircleUnpred', '02', ...
            'GarborFreqPred', '05', 'GarborFreqUnpred', '06', ...
            'GarborOrientationPred', '03', 'GarborOrientationUnpred', '04', ...
            'GarborSCPred', '07', 'GarborSCUnpred', '08');
        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    end

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    %% Process Training Data
    fmri_train_pred = [];
    fmri_train_unpred = [];

    for date_idx = 1:numel(train_dates)
        train_date = train_dates{date_idx};
        pred_data = [];
        unpred_data = [];

        for task_idx = 1:numel(train_tasks)
            task_name = train_tasks{task_idx};
            dynamic_filename = fullfile(data_path, [train_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
            disp(['Loading training file: ', dynamic_filename]);

            if exist(dynamic_filename, 'file')
                load(dynamic_filename, 'stimulus_only_data');
                data = double(stimulus_only_data); % Extract stimulus-only data
                
                % Concatenate based on condition (Pred/Unpred)
                if contains(task_name, 'Pred')
                    pred_data = [pred_data, data];
                elseif contains(task_name, 'Unpred')
                    unpred_data = [unpred_data, data];
                end
            else
                warning(['File not found: ', dynamic_filename]);
            end
        end

        % Concatenate across dates for train
        if ~isempty(pred_data)
            fmri_train_pred = cat(3, fmri_train_pred, pred_data);
        end
        if ~isempty(unpred_data)
            fmri_train_unpred = cat(3, fmri_train_unpred, unpred_data);
        end
    end

    % Save concatenated training data
    % train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);
    % save(fullfile(save_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_Pred.mat']), 'fmri_train_pred', '-v7.3');
    % save(fullfile(save_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_Unpred.mat']), 'fmri_train_unpred', '-v7.3');

    disp('Saved concatenated training data for Pred and Unpred.');

    %% Process Testing Data
    fmri_test_pred = [];
    fmri_test_unpred = [];

    for task_idx = 1:numel(train_tasks)
        task_name = train_tasks{task_idx};
        dynamic_filename = fullfile(data_path, [test_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
        disp(['Loading testing file: ', dynamic_filename]);

        if exist(dynamic_filename, 'file')
            load(dynamic_filename, 'stimulus_only_data');
            data = double(stimulus_only_data); % Extract stimulus-only data

            % Concatenate based on condition (Pred/Unpred)
            if contains(task_name, 'Pred')
                fmri_test_pred = [fmri_test_pred, data];
            elseif contains(task_name, 'Unpred')
                fmri_test_unpred = [fmri_test_unpred, data];
            end
        else
            warning(['File not found: ', dynamic_filename]);
        end
    end

    % Save concatenated testing data
    save(fullfile(save_path, ['testing_fmri_', test_date(1:6), '_Pred.mat']), 'fmri_test_pred', '-v7.3');
    save(fullfile(save_path, ['testing_fmri_', test_date(1:6), '_Unpred.mat']), 'fmri_test_unpred', '-v7.3');

    disp('Saved concatenated testing data for Pred and Unpred.');
end

disp('All configurations processed successfully.');


%% Movie task
%% Parameters
Nv = 91282; % Number of voxels

save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding_concat_time_noHRF/movie/';
data_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/noHRF/';

% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1';
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Default exclude tasks
exclude_tasks = {'GarborPhasePred', 'GarborPhaseUnpred',...
    'CirclePred', 'CircleUnpred', ...
    'GarborFreqPred', 'GarborFreqUnpred',...
    'GarborOrientationPred', 'GarborOrientationUnpred',...
    'GarborSCPred', 'GarborSCUnpred'}; % Exclude garbor tasks

%% Process fMRI Data for Each Configuration
for config_idx = 1:size(configs, 1)
    train_dates = configs{config_idx, 1};
    test_date = configs{config_idx, 2};

    % Dynamically assign run_numbers based on scanning_dates
    if ismember('241119_noT1', train_dates)
        run_numbers = struct('MoviePred', '11', 'MovieUnpred', '12');
        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    else
        run_numbers = struct('MoviePred', '09', 'MovieUnpred', '10');
        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    end

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    %% Process Training Data
    fmri_train_pred = [];
    fmri_train_unpred = [];

    for date_idx = 1:numel(train_dates)
        train_date = train_dates{date_idx};
        pred_data = [];
        unpred_data = [];

        for task_idx = 1:numel(train_tasks)
            task_name = train_tasks{task_idx};
            dynamic_filename = fullfile(data_path, [train_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
            disp(['Loading training file: ', dynamic_filename]);

            if exist(dynamic_filename, 'file')
                load(dynamic_filename, 'stimulus_only_data');
                data = double(stimulus_only_data); % Extract stimulus-only data
                
                % Concatenate based on condition (Pred/Unpred)
                if contains(task_name, 'Pred')
                    pred_data = [pred_data, data];
                elseif contains(task_name, 'Unpred')
                    unpred_data = [unpred_data, data];
                end
            else
                warning(['File not found: ', dynamic_filename]);
            end
        end

        % Concatenate across dates for train
        if ~isempty(pred_data)
            fmri_train_pred = cat(3, fmri_train_pred, pred_data);
        end
        if ~isempty(unpred_data)
            fmri_train_unpred = cat(3, fmri_train_unpred, unpred_data);
        end
    end

    % Save concatenated training data
    train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);
    save(fullfile(save_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_Pred.mat']), 'fmri_train_pred', '-v7.3');
    save(fullfile(save_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_Unpred.mat']), 'fmri_train_unpred', '-v7.3');

    disp('Saved concatenated training data for Pred and Unpred.');

    %% Process Testing Data
    fmri_test_pred = [];
    fmri_test_unpred = [];

    for task_idx = 1:numel(train_tasks)
        task_name = train_tasks{task_idx};
        dynamic_filename = fullfile(data_path, [test_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
        disp(['Loading testing file: ', dynamic_filename]);

        if exist(dynamic_filename, 'file')
            load(dynamic_filename, 'stimulus_only_data');
            data = double(stimulus_only_data); % Extract stimulus-only data

            % Concatenate based on condition (Pred/Unpred)
            if contains(task_name, 'Pred')
                fmri_test_pred = [fmri_test_pred, data];
            elseif contains(task_name, 'Unpred')
                fmri_test_unpred = [fmri_test_unpred, data];
            end
        else
            warning(['File not found: ', dynamic_filename]);
        end
    end

    % Save concatenated testing data
    save(fullfile(save_path, ['testing_fmri_', test_date(1:6), '_Pred.mat']), 'fmri_test_pred', '-v7.3');
    save(fullfile(save_path, ['testing_fmri_', test_date(1:6), '_Unpred.mat']), 'fmri_test_unpred', '-v7.3');

    disp('Saved concatenated testing data for Pred and Unpred.');
end

disp('All configurations processed successfully.');