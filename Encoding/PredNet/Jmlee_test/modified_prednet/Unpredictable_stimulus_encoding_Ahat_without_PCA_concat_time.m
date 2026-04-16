clc; clear;

%% calculate the temporal mean and standard deviation of the feature time series of CNN units
% CNN layer labels
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3';};

stimulus_type = {'circle','garbor_frequency',  'garbor_orientation', 'garbor_sc'};
session_type = 'unpredictable_30fps';


% 128*160 img size
base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Unpredictable_stimulus/layer4/Ahat/garbor/';

disp('Start mean calculation');
% Calculate the temporal mean
for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Path to the H5 file
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/layer4/');
        secpath = fullfile(dataroot, 'Ahat.h5');

        if exist(secpath, 'file')
            lay_feat = h5read(secpath, [layername{lay}]);
            dim = size(lay_feat);

            % Calculate the mean
            lay_feat_mean = mean(lay_feat, ndims(lay_feat));

            % Save the mean for the current stimulus
            disp(['Saving mean for layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);
            h5create([saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], size(lay_feat_mean), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], lay_feat_mean);
        else
            disp(['File not found: ', secpath]);
        end
    end
end
disp('Mean calculation completed.');

disp('Start standard deviation calculation');
% Calculate the temporal standard deviation
for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Path to the H5 file
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/');
        secpath = fullfile(dataroot, 'Ahat.h5');
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];

        if exist(secpath, 'file') && exist(mean_path, 'file')
            % Read the feature map and mean for the current layer
            lay_feat = h5read(secpath, layername{lay});
            lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);

            % Calculate standard deviation
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat_std = sqrt(mean(lay_feat.^2, ndims(lay_feat)));
            lay_feat_std(lay_feat_std == 0) = 1;

            % Save the standard deviation for the current stimulus
            disp(['Saving standard deviation for layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);
            h5create([saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], size(lay_feat_std), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], lay_feat_std);
        else
            disp(['File not found or mean missing: ', secpath, ' or ', mean_path]);
        end
    end
end
disp('Standard deviation calculation completed.');

% Process and Convolve with HRF
disp('Start processing feature maps');
% HRF configuration
srate = 30; % Sampling rate for simple stimuli
p = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1 / srate, p);
hrf = hrf(:);

for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Processing Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Load precomputed mean and std
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];
        std_path = [saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'];
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/layer4/');
        secpath = fullfile(dataroot, 'Ahat.h5');

        if exist(secpath, 'file') && exist(mean_path, 'file') && exist(std_path, 'file')
            lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);
            lay_feat_std = h5read(std_path, [layername{lay}, '/data']);
            lay_feat = h5read(secpath, [layername{lay}]);

            % Standardize the feature map
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0;

            % Reshape feature map for processing
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)), dim(end)); % Reshape for convolution

            % Convolve with HRF
            disp('Convolve with HRF');
            ts = conv2(hrf, lay_feat', 'full');
            start_idx = 4 * srate + 1; % Adjust for HRF peak
            ts = ts(start_idx:start_idx + 154, :); % Adjust for fMRI timepoints (155)

            % Save processed feature map
            disp('Saving processed feature map');
            reduced_path = [saveroot, 'PredNet_feature_maps_processed_', stimulus_type{stim}, '_layer', num2str(lay), '.h5'];
            h5create(reduced_path, [layername{lay}, '/data'], size(ts), 'Datatype', 'single');
            h5write(reduced_path, [layername{lay}, '/data'], ts);
        else
            disp(['Missing data for stimulus: ', stimulus_type{stim}]);
        end
    end
end
disp('Feature map processing completed.');


%% Movie task 
clc; clear;

% Calculate the temporal mean and standard deviation of the feature time series of CNN units
% CNN layer labels
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3';};

stimulus_type = {'movie'};
session_type = 'unpredictable_20fps';

% Base directory and save root
base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/modified_prednet/concat_time/Unpredictable_stimulus/layer4/Ahat/movie/';

disp('Start mean calculation');
% Calculate the temporal mean
for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Path to the H5 file
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/layer4/');
        secpath = fullfile(dataroot, 'Ahat.h5');

        if exist(secpath, 'file')
            lay_feat = h5read(secpath, [layername{lay}]);
            dim = size(lay_feat);

            % Calculate the mean
            lay_feat_mean = mean(lay_feat, ndims(lay_feat));

            % Save the mean for the current stimulus
            disp(['Saving mean for layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);
            h5create([saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], size(lay_feat_mean), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], lay_feat_mean);
        else
            disp(['File not found: ', secpath]);
        end
    end
end
disp('Mean calculation completed.');

disp('Start standard deviation calculation');
% Calculate the temporal standard deviation
for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Path to the H5 file
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/layer4/');
        secpath = fullfile(dataroot, 'Ahat.h5');
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];

        if exist(secpath, 'file') && exist(mean_path, 'file')
            % Read the feature map and mean for the current layer
            lay_feat = h5read(secpath, layername{lay});
            lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);

            % Calculate standard deviation
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat_std = sqrt(mean(lay_feat.^2, ndims(lay_feat)));
            lay_feat_std(lay_feat_std == 0) = 1;

            % Save the standard deviation for the current stimulus
            disp(['Saving standard deviation for layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);
            h5create([saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], size(lay_feat_std), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'], [layername{lay}, '/data'], lay_feat_std);
        else
            disp(['File not found or mean missing: ', secpath, ' or ', mean_path]);
        end
    end
end
disp('Standard deviation calculation completed.');

% Process and Convolve with HRF
disp('Start processing feature maps');
% HRF configuration
srate = 20; % Sampling rate for simple stimuli
p = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1 / srate, p);
hrf = hrf(:);

for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Processing Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Load precomputed mean and std
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];
        std_path = [saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'];
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_CNN_like_ver2_A_pass/h5/layer4/');
        secpath = fullfile(dataroot, 'Ahat.h5');

        if exist(secpath, 'file') && exist(mean_path, 'file') && exist(std_path, 'file')
            lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);
            lay_feat_std = h5read(std_path, [layername{lay}, '/data']);
            lay_feat = h5read(secpath, [layername{lay}]);

            % Standardize the feature map
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0;

            % Reshape feature map for processing
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)), dim(end)); % Reshape for convolution

            % Convolve with HRF
            disp('Convolve with HRF');
            ts = conv2(hrf, lay_feat', 'full');
            start_idx = 4 * srate + 1; % Adjust for HRF peak
            ts = ts(start_idx:start_idx + 154, :); % Adjust for fMRI timepoints (155)

            % Save processed feature map
            disp('Saving processed feature map');
            reduced_path = [saveroot, 'PredNet_feature_maps_processed_', stimulus_type{stim}, '_layer', num2str(lay), '.h5'];
            h5create(reduced_path, [layername{lay}, '/data'], size(ts), 'Datatype', 'single');
            h5write(reduced_path, [layername{lay}, '/data'], ts);
        else
            disp(['Missing data for stimulus: ', stimulus_type{stim}]);
        end
    end
end
disp('Feature map processing completed.');
