clc; clear;

%% Parameters
Nv = 91282; % Number of voxels

save_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/for_encoding/';
data_path = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/fmri/noHRF/';
% Configurations for training and testing
configs = {
    {'241119_noT1', '241121_noT1'}, '241205_T1';
    {'241119_noT1', '241205_T1'}, '241121_noT1';
    {'241121_noT1', '241205_T1'}, '241119_noT1';
};

% Default exclude tasks
exclude_tasks = {'GarborPhasePred', 'GarborPhaseUnpred'}; % Only for 241119_noT1

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
            'GarborSCPred', '07', 'GarborSCUnpred', '08', ...
            'MoviePred', '11', 'MovieUnpred', '12');
        train_tasks = setdiff(fieldnames(run_numbers), exclude_tasks);
    else
        run_numbers = struct(...
            'CirclePred', '01', 'CircleUnpred', '02', ...
            'GarborFreqPred', '05', 'GarborFreqUnpred', '06', ...
            'GarborOrientationPred', '03', 'GarborOrientationUnpred', '04', ...
            'GarborSCPred', '07', 'GarborSCUnpred', '08', ...
            'MoviePred', '09', 'MovieUnpred', '10');
        train_tasks = fieldnames(run_numbers);
    end

    disp(['Training on: ', strjoin(train_dates, ', ')]);
    disp(['Testing on: ', test_date]);

    % Process Training Data
    for task_idx = 1:numel(train_tasks)
        task_name = train_tasks{task_idx};
        fmri_train = [];
        train_idx = 1;

        disp(['Processing training data for task: ', task_name]);
        for date_idx = 1:numel(train_dates)
            train_date = train_dates{date_idx};

            % File paths for ITI-removed data
            dynamic_filename = fullfile(data_path, [train_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
            disp(['Loading training file: ', dynamic_filename]);

            if exist(dynamic_filename, 'file')
                load(dynamic_filename, 'stimulus_only_data');
                Nt = size(stimulus_only_data, 2);

                % Initialize fmri_train after determining Nt
                if isempty(fmri_train)
                    fmri_train = zeros(Nv, Nt, numel(train_dates), 'single');
                end

                data = double(stimulus_only_data); % Extract stimulus-only data
                
                % Standardization
                data = bsxfun(@minus, data, mean(data, 2));
                data = bsxfun(@rdivide, data, std(data, [], 2));
                fmri_train(:, :, train_idx) = data;
                train_idx = train_idx + 1;
            else
                warning(['File not found: ', dynamic_filename]);
            end
        end

        % Save training data for this task
        train_dates_clean = cellfun(@(d) d(1:6), train_dates, 'UniformOutput', false);
        train_file = fullfile(save_path, ['training_fmri_', strjoin(train_dates_clean, '_'), '_', task_name, '.mat']);
        save(train_file, 'fmri_train', '-v7.3');
        disp(['Saved training data for task ', task_name, ': ', train_file]);
    end

    % Process Testing Data
    for task_idx = 1:numel(train_tasks)
        task_name = train_tasks{task_idx};
        fmri_test = [];
        test_idx = 1;

        disp(['Processing testing data for task: ', task_name]);
        dynamic_filename = fullfile(data_path, [test_date(1:6), '_', task_name, '_noITI_noHRF.mat']);
        disp(['Loading testing file: ', dynamic_filename]);

        if exist(dynamic_filename, 'file')
            load(dynamic_filename, 'stimulus_only_data');
            Nt = size(stimulus_only_data, 2);

            % Initialize fmri_test after determining Nt
            if isempty(fmri_test)
                fmri_test = zeros(Nv, Nt, 1, 'single');
            end

            data = double(stimulus_only_data); % Extract stimulus-only data
            
            % Standardization
            data = bsxfun(@minus, data, mean(data, 2));
            data = bsxfun(@rdivide, data, std(data, [], 2));
            fmri_test(:, :, test_idx) = data;
        else
            warning(['File not found: ', dynamic_filename]);
        end

        % Save testing data for this task
        test_file = fullfile(save_path, ['testing_fmri_', test_date(1:6), '_', task_name, '.mat']);
        save(test_file, 'fmri_test', '-v7.3');
        disp(['Saved testing data for task ', task_name, ': ', test_file]);
    end
end

disp('All configurations processed successfully.');
