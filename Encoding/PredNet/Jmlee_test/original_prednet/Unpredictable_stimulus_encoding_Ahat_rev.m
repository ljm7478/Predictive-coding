clc;clear

%% calculate the temporal mean and standard deviation of the feature time series of CNN units
% CNN layer labels
layername = {'/E0';'/E1';'/E2';'/E3';}; 

% stimulus_type = {'circle','garbor_frequency',  'garbor_orientation', 'garbor_sc'}; 
% session_type = 'unpredictable_30fps';

stimulus_type = {'movie'};
session_type = 'unpredictable_20fps';

%128*160 img size
base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp' ; 
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/original_prednet/E/';

disp('start mean')
% calculate the temporal mean 
for lay = 1: length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Path to the H5 file
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_original/h5/');
        secpath = fullfile(dataroot, 'E.h5');

        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay}]);
            dim = size(lay_feat) % convolutional layers: #kernel*fmsize1*fmsize2*#frames

            lay_feat_mean = zeros([dim(1:end-1),1]);
            
            % Accumulate the sum of features and update frame count
            lay_feat_mean = sum(lay_feat, length(dim)) / dim(end);

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
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_original/h5/');
        secpath = fullfile(dataroot, 'E.h5');
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];

        if exist(secpath, 'file') && exist(mean_path, 'file')
            % Read the feature map and mean for the current layer
            lay_feat = h5read(secpath, layername{lay});
            lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);

            % Calculate standard deviation
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = lay_feat.^2;
            dim = size(lay_feat);

            lay_feat_std = sqrt(sum(lay_feat, length(dim)) / (dim(end) - 1));
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



%% Reduce the dimension of the PredNet features for encoding models - %128*160 img size 
% Dimension reduction by using principal component analysis (PCA)
% Here provides two ways to compute the PCAs. If the memory is big enough
% to directly calculate the svd, then use the method 1, otherwise use the
% SVD-updating algorithm (Zha and Simon, 1999; Zhao et al., 2006; Wen et al. 2017).

% % % % % % % % % Direct SVD % % % % % % % % % % % % % %
% This requires big memory to compute. It's better to reduce the
% frame rate of videos or use the SVD-updating algorithm.
layername = {'/E0';'/E1';'/E2';'/E3';}; 
% 
stimulus_type = {'circle','garbor_frequency', 'garbor_orientation', 'garbor_sc'};   
session_type = 'unpredictable_30fps';

% stimulus_type = {'movie'};
% session_type = 'unpredictable_20fps';

% 128*160 img size
base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp'; 
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/original_prednet/E/';

for lay = 1:length(layername)
    for stim = 1:length(stimulus_type)
        disp(['Processing PCA for Layer: ', layername{lay}, '; Stimulus: ', stimulus_type{stim}]);

        % Load mean and std
        mean_path = [saveroot, 'PredNet_feature_maps_avg_', stimulus_type{stim}, '.h5'];
        std_path = [saveroot, 'PredNet_feature_maps_std_', stimulus_type{stim}, '.h5'];
        if ~isfile(mean_path) || ~isfile(std_path)
            disp(['Missing files for stimulus: ', stimulus_type{stim}]);
            continue;
        end

        lay_feat_mean = h5read(mean_path, [layername{lay}, '/data']);
        lay_feat_std = h5read(std_path, [layername{lay}, '/data']);
        lay_feat_mean = lay_feat_mean(:);
        lay_feat_std = lay_feat_std(:);

        % Load feature map
        dataroot = fullfile(base_dir, stimulus_type{stim}, session_type, 'result/prednet_original/h5/');
        secpath = fullfile(dataroot, 'E.h5');
        if ~isfile(secpath)
            disp(['Feature map file missing: ', secpath]);
            continue;
        end

        lay_feat = h5read(secpath, [layername{lay}]);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat, Nu, Nf); % Nu x Nf

        % Standardize the feature map
        disp('Start standardization');
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0;

        % Perform PCA
        disp('Performing PCA');
        if size(lay_feat, 1) > size(lay_feat, 2)
            R = lay_feat' * lay_feat / size(lay_feat, 1);
            [U, S] = svd(R);
            s = diag(S);

            % Keep 99% variance
            ratio = cumsum(s) / sum(s);
            Nc = find(ratio > 0.99, 1, 'first');

            S_2 = diag(1 ./ sqrt(s(1:Nc)));
            B = lay_feat * (U(:, 1:Nc) * S_2 / sqrt(size(lay_feat, 1)));
        else
            R = lay_feat * lay_feat';
            [U, S] = svd(R);
            s = diag(S);

            % Keep 99% variance
            ratio = cumsum(s) / sum(s);
            Nc = find(ratio > 0.99, 1, 'first');

            B = U(:, 1:Nc);
        end

        % Save principal components
        disp('Start saving PCA results');
        save_filename = [saveroot, 'PredNet_feature_maps_pca_', stimulus_type{stim}, '_layer', num2str(lay), '.mat'];
        save(save_filename, 'B', 's', '-v7.3');
        disp(['PCA results saved: ', save_filename]);
    end
end
disp('FINISH PCA');


%% Processed the dimension-reduced 
clc;clear

layername = {'/E0';'/E1';'/E2';'/E3';}; 

stimulus_type = {'garbor_frequency'}; %{'circle', 'garbor_orientation', 'garbor_sc'};  %{'garbor_frequency'}
session_type = 'unpredictable_30fps';

stimulus_type_movie = {'movie'};
session_type_movie = 'unpredictable_20fps';

%128*160 img size
base_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_jmlee_stimuli/PredNetExp'; 
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Jmlee_test/encoding_analysis/original_prednet/E/';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
% srate = 30; %zhongmin liu
% srate = 25; %gump
srate = 30; % jmlee_stimulus - simple (circle, garbor) - 1TR
srate_movie = 20; % jmlee_stimulus - movie - 1TR

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.

p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate, p);
hrf_movie = spm_hrf(1/srate_movie, p);
hrf = hrf(:);
hrf_movie = hrf_movie(:);
% figure; plot(0:1/srate:p(7),hrf);

disp('start dimesion reduction')
% sessions = {stimulus_type, session_type, srate, hrf; stimulus_type_movie, session_type_movie, srate_movie, hrf_movie};
sessions = {stimulus_type_movie, session_type_movie, srate_movie, hrf_movie};


for session_idx = 1:size(sessions, 1)
    curr_stimulus_type = sessions{session_idx, 1};
    curr_session_type = sessions{session_idx, 2};
    curr_srate = sessions{session_idx, 3};
    curr_hrf = sessions{session_idx, 4};
    
    for lay = 1: length(layername)
        % Dimension reduction for training data
        for stim = 1:length(curr_stimulus_type)
            disp(['Processing Layer: ', layername{lay}, '; Stimulus: ', curr_stimulus_type{stim}]);

            % Load precomputed mean, std, and PCA matrix
            % lay_feat_mean = h5read([saveroot, 'PredNet_feature_maps_avg_', curr_stimulus_type{stim}, '.h5'], [layername{lay}, '/data']);
            % lay_feat_std = h5read([saveroot, 'PredNet_feature_maps_std_', curr_stimulus_type{stim}, '.h5'], [layername{lay}, '/data']);

            lay_feat_mean = h5read([saveroot, 'PredNet_feature_maps_avg_garbor.h5'], [layername{lay}, '/data']);
            lay_feat_std = h5read([saveroot, 'PredNet_feature_maps_std_garbor.h5'], [layername{lay}, '/data']);
            load([saveroot, 'PredNet_feature_maps_pca_',  curr_stimulus_type{stim}, '_layer', num2str(lay), '.mat'], 'B');
            % load([saveroot,'PredNet_feature_maps_svd_layer',num2str(lay),'.mat'], 'B');

            dataroot = fullfile(base_dir, curr_stimulus_type{stim}, curr_session_type, 'result/prednet_original/h5/');
            secpath = fullfile(dataroot, 'E.h5');
            if exist(secpath,'file')==2
                lay_feat = h5read(secpath,[layername{lay}]);

                % Standardize the feature map
                lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
                lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
                lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

                % Reshape feature map
                dim = size(lay_feat);
                lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

                % Apply PCA
                Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components

                ts = conv2(curr_hrf, Y); % Convolve with HRF

                % Align the time series with HRF
                start_idx = 4 * curr_srate + 1; % Adjust for HRF peak

                required_length = 155; % Number of fMRI timepoints required
                end_idx = start_idx + required_length - 1;

                % Debugging information
                disp(['Start index: ', num2str(start_idx)]);
                disp(['End index: ', num2str(end_idx)]);

                % Ensure ts has enough data
                if size(ts, 1) < end_idx
                    % Pad with zeros if shorter
                    ts = [ts(start_idx:end, :); zeros(end_idx - size(ts, 1), size(ts, 2))];
                elseif size(ts, 1) >= end_idx
                    % Trim if longer or exactly matches the required length
                    ts = ts(start_idx:end_idx, :);
                end

                % end_idx = start_idx + 155 * curr_srate / 30 - 1; % Match to 155 fMRI timepoints
                % if end_idx > size(ts, 1)
                %     % Trim if ts is longer than needed
                %     ts = ts(start_idx:end, :);
                % elseif size(ts, 1) < end_idx
                %     % Pad with zeros if ts is shorter than 155
                %     ts = [ts(start_idx:end, :); zeros(end_idx - size(ts, 1), size(ts, 2))];
                % else
                %     % Trim to exactly 155 points
                %     ts = ts(start_idx:end_idx, :);
                % end

                % Downsample if needed (1TR = 1 second)
                if size(ts, 1) > 155
                    ts = ts(1:155, :); % Trim extra frames
                end

                disp('start saving dimesion reduction')
                
                reduced_path = [saveroot, 'PredNet_feature_maps_pcareduced_', curr_stimulus_type{stim}, '_layer', num2str(lay), '.h5'];
                disp('Saving dimension-reduced features');
                h5create(reduced_path, [layername{lay}, '/data'], size(ts), 'Datatype', 'single');
                h5write(reduced_path, [layername{lay}, '/data'], ts);

                % h5create([saveroot,'PredNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
                %     [size(ts)],'Datatype','single');
                % h5write([saveroot,'PredNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
            end
        end   
        disp('finish train ts')
    end 
end

