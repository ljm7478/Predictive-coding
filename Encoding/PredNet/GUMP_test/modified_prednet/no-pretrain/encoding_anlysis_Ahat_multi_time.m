%% mean and std of training runs (from run01-04)
clc; clear;

% Define paths
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_CNN_like_ver2_A_pass/h5/layer4/multi_prediction/2second_50frame';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/multi_time_result/2second_50frame/layer4/Ahat/';
if ~exist(saveroot, 'dir')
    mkdir(saveroot); % Create the directory if it does not exist
end

% CNN layer labels
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'}; 

% Frame counts for each run (training: run1-4, testing: run5-8)
frame_counts = [22550, 22050, 21900, 24400, 23100, 21950, 27100, 16876];

% Number of training and testing runs
num_train = 4;  
num_test = 4;   

frame_counts = [22550, 22050, 21900, 24400, 23100, 21950, 27100, 16876];

% Stride settings
is_stride = true;    % false = 25 fps, true = stride subsampling
stride    = 50;      % number of frames per step (1s at 25 fps)

if is_stride
    frame_counts = round(frame_counts / stride);
end

disp('Starting computation of mean and standard deviation');

% **Compute the temporal mean across training runs**
for lay = 1:length(layername)
    disp(['Processing mean for layer: ', layername{lay}]);

    N = 0;
    lay_feat_sum = [];

    for seg = 0:num_train
        secpath = fullfile(dataroot, ['seg', num2str(seg), '_Ahat.h5']);
        if exist(secpath, 'file')
            lay_feat = h5read(secpath, [layername{lay}]);  
            dim = size(lay_feat);

            % Initialize sum accumulator on first seg
            if isempty(lay_feat_sum)
                lay_feat_sum = zeros([dim(1:end-1), 1]);
            end

            % Sum across frames
            lay_feat_sum = lay_feat_sum + sum(lay_feat, length(dim));
            if frame_counts(seg+1) == dim(end)
                N = N + frame_counts(seg+1);
            end
        else
            warning(['Missing file: ', secpath]);
        end
    end

    % Compute mean
    lay_feat_mean = lay_feat_sum / N;

    % Save mean
    h5create([saveroot, 'PredNet_feature_maps_avg.h5'], [layername{lay}, '/data'], size(lay_feat_mean), 'Datatype', 'single');
    h5write([saveroot, 'PredNet_feature_maps_avg.h5'], [layername{lay}, '/data'], lay_feat_mean);
end

disp('Mean computation completed.');

% **Compute standard deviation**
disp('Starting standard deviation computation');

for lay = 1:length(layername)
    disp(['Processing std for layer: ', layername{lay}]);

    % Load mean
    lay_feat_mean = h5read([saveroot, 'PredNet_feature_maps_avg.h5'], [layername{lay}, '/data']);
    N = 0;
    lay_feat_sqsum = [];

    for seg = 0:num_train
        secpath = fullfile(dataroot, ['seg', num2str(seg), '_Ahat.h5']);
        if exist(secpath, 'file')
            lay_feat = h5read(secpath, [layername{lay}]);  

            % Compute squared difference
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = lay_feat .^ 2;
            dim = size(lay_feat);

            if isempty(lay_feat_sqsum)
                lay_feat_sqsum = zeros([dim(1:end-1), 1]);
            end

            % Sum squared differences
            lay_feat_sqsum = lay_feat_sqsum + sum(lay_feat, length(dim));
            if frame_counts(seg+1) == dim(end)
                N = N + frame_counts(seg+1);
            end
        else
            warning(['Missing file: ', secpath]);
        end
    end

    % Compute std
    lay_feat_std = sqrt(lay_feat_sqsum / (N - 1));
    lay_feat_std(lay_feat_std == 0) = 1;

    % Save standard deviation
    h5create([saveroot, 'PredNet_feature_maps_std.h5'], [layername{lay}, '/data'], size(lay_feat_std), 'Datatype', 'single');
    h5write([saveroot, 'PredNet_feature_maps_std.h5'], [layername{lay}, '/data'], lay_feat_std);
end

disp('Standard deviation computation completed.');

%% **Perform PCA on Training Data**
disp('Starting PCA computation');

for lay = 1:length(layername)
    disp(['Processing PCA for layer: ', layername{lay}]);

    % Load mean and std
    lay_feat_mean = h5read([saveroot, 'PredNet_feature_maps_avg.h5'], [layername{lay}, '/data']);
    lay_feat_std = h5read([saveroot, 'PredNet_feature_maps_std.h5'], [layername{lay}, '/data']);
    lay_feat_mean = lay_feat_mean(:);
    lay_feat_std = lay_feat_std(:);

     % Collect training data from all runs
    lay_feat_cell = cell(1, num_train);

    for seg = 0:num_train
        secpath = fullfile(dataroot, ['seg', num2str(seg), '_Ahat.h5']);
        if exist(secpath, 'file')
            lay_feat = h5read(secpath, layername{lay});
            dim = size(lay_feat);
            Nu = prod(dim(1:end-1));  % Number of units
            Nf = dim(end);  % Number of frames
            lay_feat = reshape(lay_feat, Nu, Nf);

            lay_feat_cell{seg+1} = lay_feat;
        else
            warning(['Missing file: ', secpath]);
            lay_feat_cell{seg+1} = [];
        end
    end

    % Concatenate all runs
    lay_feat_cont = cat(2, lay_feat_cell{:});

    % Standardize
    lay_feat_cont = bsxfun(@minus, lay_feat_cont, lay_feat_mean);
    lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std);
    lay_feat_cont(isnan(lay_feat_cont)) = 0;

    % Compute PCA
    if size(lay_feat_cont,1) > size(lay_feat_cont,2)
        R = lay_feat_cont'*lay_feat_cont/size(lay_feat_cont,1);
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        S_2 = diag(1./sqrt(s(1:Nc))); % S.^(-1/2)
        B = lay_feat_cont*(U(:,1:Nc)*S_2/sqrt(size(lay_feat_cont,1)));
        % I = B'*B; % check if I is an indentity matrix
        
    else
        R = lay_feat_cont*lay_feat_cont';
        [U,S] = svd(R);
        s = diag(S);
        
        % keep 99% variance
        ratio = cumsum(s)/sum(s); 
        Nc = find(ratio>0.99,true,'first'); % number of components
        
        B = U(:,1:Nc);
    end

    % save principal components
    save([saveroot,'PredNet_feature_maps_pca_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');
end
disp('PCA computation completed.');


%% Process train data: Apply PCA, convolve with HRF, and downsample to match fMRI
disp('Starting test data processing...');

% ---- PARAMETERS YOU CONTROL PER DATASET ----
is_stride = true;   % false for 25 fps t+1; true for stride=25 (1s samples)
stride    = 50;     % used only if is_stride = true
TR        = 2.0;    % your fMRI TR in seconds

% ---- EFFECTIVE SAMPLING RATE OF FEATURES ----
if is_stride
    srate = 25/stride;    % = 1 Hz for stride=25
else
    srate = 25;           % 25 Hz for original t+1 features
end

% ----for 2second only before the layer/seg loops ----
step = round(TR * srate);   % samples per fMRI volume
if step < 1
    step = 1;
end

% Define sampling rate
p = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate, p);
hrf = hrf(:);  % Define HRF

for lay = 1:length(layername)
    disp(['Processing test data for layer: ', layername{lay}]);

    % Load mean, std, and PCA matrix
    lay_feat_mean = h5read([saveroot, 'PredNet_feature_maps_avg.h5'], [layername{lay}, '/data']);
    lay_feat_std = h5read([saveroot, 'PredNet_feature_maps_std.h5'], [layername{lay}, '/data']);
    load([saveroot, 'PredNet_feature_maps_pca_layer', num2str(lay), '.mat'], 'B');

    for seg = 0:3  % Train runs (run1 - run4)
        disp(['Processing seg: ', num2str(seg+1)]);
        secpath = fullfile(dataroot, ['seg', num2str(seg), '_Ahat.h5']);
        
        if exist(secpath, 'file')
            % Load and standardize test features
            lay_feat = h5read(secpath, [layername{lay}]);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0;

            % Reshape and apply PCA
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)), dim(end));
            Y = lay_feat' * B / sqrt(size(B, 1));  % PCA projection

            % Convolve with HRF
            ts = conv2(hrf, Y);  
            start_idx = 4 * srate + 1;  % Adjust for HRF peak
            end_idx = start_idx + size(Y, 1) - 1;

            if end_idx > size(ts, 1)
                ts = ts(start_idx:end, :);
            else
                ts = ts(start_idx:end_idx, :);
            end

            % Downsample to match fMRI (2 Hz)
            % ts = ts(srate+1:2*srate:end, :);

            % TR-aware downsampling for 2second only
            ts = ts(1:step:end, :);

            disp(['ts shape is : ', num2str(size(ts))])


            % Save PCA-reduced test features
            h5create([saveroot, 'PredNet_feature_maps_pcareduced_run', num2str(seg+1), '.h5'], [layername{lay}, '/data'], size(ts), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_pcareduced_run', num2str(seg+1), '.h5'], [layername{lay}, '/data'], ts);
        else
            warning(['Test file missing: ', secpath]);
        end
    end

    for seg = 4:7  % Test runs (run5 - run8)
        disp(['Processing seg: ', num2str(seg+1)]);
        secpath = fullfile(dataroot, ['seg', num2str(seg), '_Ahat.h5']);
        
        if exist(secpath, 'file')
            % Load and standardize test features
            lay_feat = h5read(secpath, [layername{lay}]);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0;

            % Reshape and apply PCA
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)), dim(end));
            Y = lay_feat' * B / sqrt(size(B, 1));  % PCA projection

            % Convolve with HRF
            ts = conv2(hrf, Y);  
            start_idx = 4 * srate + 1;  % Adjust for HRF peak
            end_idx = start_idx + size(Y, 1) - 1;

            if end_idx > size(ts, 1)
                ts = ts(start_idx:end, :);
            else
                ts = ts(start_idx:end_idx, :);
            end

            % Downsample to match fMRI (2 Hz)
            % ts = ts(srate+1:2*srate:end, :);

            % TR-aware downsampling for 2second only
            ts = ts(1:step:end, :);

            disp(['ts shape is : ', num2str(size(ts))])


            % Save PCA-reduced test features
            h5create([saveroot, 'PredNet_feature_maps_pcareduced_run', num2str(seg+1), '.h5'], [layername{lay}, '/data'], size(ts), 'Datatype', 'single');
            h5write([saveroot, 'PredNet_feature_maps_pcareduced_run', num2str(seg+1), '.h5'], [layername{lay}, '/data'], ts);
        else
            warning(['Test file missing: ', secpath]);
        end
    end
end

disp('Test data processing completed.');

%% Concatenate the dimension-reduced CNN features of training movies
% CNN layer labels
layername = {'/Ahat0'; '/Ahat1'; '/Ahat2'; '/Ahat3'}; 
dataroot = saveroot; %'/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/layer4/Ahat/';
fmri_volumes = [451, 441, 438, 488];

for lay = 1 : length(layername)
    disp(['Processing Layer: ', layername{lay}]);

    % Initialize an empty cell array to store feature data for concatenation
    train_feat = cell(1, 4);
    for seg = 0 : 3
        disp(['Layer: ', num2str(lay),'; seg: ',num2str(seg)]);
        secpath = [dataroot,'PredNet_feature_maps_pcareduced_run', num2str(seg+1),'.h5']; 
         if exist(secpath, 'file')
            % Load feature data
            lay_feat = h5read(secpath, [layername{lay}, '/data']);

            % Ensure consistency in frame size
            if size(lay_feat, 1) ~= fmri_volumes(seg+1)
                warning(['Mismatch in frame count for seg ', num2str(seg+1)]);
            end

            % Store in the cell array for later concatenation
            train_feat{seg+1} = lay_feat;
        else
            warning(['Missing feature file for seg ', num2str(seg+1)]);
        end
    end

    % Concatenate training data along the **temporal dimension**
    Y = cat(1, train_feat{:}); 

    h5create([dataroot,'PredNet_feature_maps_pcareduced_concatenated.h5'],[layername{lay},'/data'],...
        [size(Y)],'Datatype','single');
    h5write([dataroot,'PredNet_feature_maps_pcareduced_concatenated.h5'], [layername{lay},'/data'], Y);
end