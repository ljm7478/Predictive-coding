clc;clear

%% calculate the temporal mean and standard deviation of the feature time series of CNN units
% CNN layer labels
% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c'; 

%128*160 img size
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_Cam-CAN/result/modified_prednet/h5/layer5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/layers4_32c/';

disp('start mean')
% calculate the temporal mean 
for lay = 1: length(layername)
    disp(['Layer: ',layername{lay}]);
    N = 0;
    
    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        secpath = [dataroot, 'Ahat.h5'];  % If layername contains 'Ahat'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];

    elseif contains(layername{lay}, 'E')
        secpath = [dataroot, 'E.h5'];  % If layername contains 'E'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];

    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end
       
    if exist(secpath, 'file')
        lay_feat = h5read(secpath,[layername{lay}]);
        dim = size(lay_feat);  % convolutional layers: #kernel * fmsize1 * fmsize2 * #frames
        lay_feat_mean = mean(lay_feat, 4); % Take the mean across the 4th dimension (time)
        
        disp('start saving mean')
        h5create([saveroot,'PredNet_feature_maps_avg.h5'],[layername{lay},'/data'],...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data'], lay_feat_mean);
    end
end

disp('start std')
% calculate the temporal standard deviation
for lay = 1: length(layername)
    disp(['Layer: ',layername{lay}]);

    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        secpath = [dataroot, 'Ahat.h5'];  % If layername contains 'Ahat'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];
        lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);

    elseif contains(layername{lay}, 'E')
        secpath = [dataroot, 'E.h5'];  % If layername contains 'E'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];
        lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end


    if exist(secpath,'file')
        lay_feat = h5read(secpath,[layername{lay}]);
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean); % Detrend (subtract mean)
        lay_feat = lay_feat.^2; % Square to compute variance
        dim = size(lay_feat);

        lay_feat_var = mean(lay_feat, 4);  % Variance
        lay_feat_std = sqrt(lay_feat_var); % Standard deviation

        lay_feat_std(lay_feat_std==0) = 1; % Handle cases where std = 0 (to avoid division by zero)
        
        disp('start saving std')
        h5create([saveroot,'PredNet_feature_maps_std.h5'],[layername{lay},'/data'],...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data'], lay_feat_std);
    end
end

%% Reduce the dimension of the PredNet features for encoding models - %128*160 img size Ahat only works
% Dimension reduction by using principal component analysis (PCA)
% Here provides two ways to compute the PCAs. If the memory is big enough
% to directly calculate the svd, then use the method 1, otherwise use the
% SVD-updating algorithm (Zha and Simon, 1999; Zhao et al., 2006; Wen et al. 2017).

% % % % % % % % % Direct SVD % % % % % % % % % % % % % %
% This requires big memory to compute. It's better to reduce the
% frame rate of videos or use the SVD-updating algorithm.

% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 
layernumber = 'layers5_32c'; 

for lay = 1: length(layername)
    disp('load mean, std')
        
    % For 'E' layers, reset lay index to start from 1
    if contains(layername{lay}, 'E')
        e_index = lay - 5;  % New index for E layers
    else
        e_index = lay;  % Keep the original index for Ahat layers
    end

    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        secpath = [dataroot, 'Ahat.h5'];  % If layername contains 'Ahat'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];
        
        % mean/std 
        lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
        lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);
        lay_feat_mean = lay_feat_mean(:);
        lay_feat_std = lay_feat_std(:);

    elseif contains(layername{lay}, 'E')
        secpath = [dataroot, 'E.h5'];  % If layername contains 'E'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];
        
        % mean/std 
        lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
        lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);
        lay_feat_mean = lay_feat_mean(:);
        lay_feat_std = lay_feat_std(:);
    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end

    lay_feat = h5read(secpath,[layername{lay}]);
    dim = size(lay_feat);
    Nu = prod(dim(1:end-1)); % number of units (channels * height * width)
    Nf = dim(end); % number of frames
    lay_feat = reshape(lay_feat, Nu, Nf); % Reshape to Nu x Nf

    disp('start standardization')

    % Standardize the time series for each unit
    lay_feat_cont = bsxfun(@minus, lay_feat, lay_feat_mean);  % Subtract mean
    lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std); % Divide by std
    lay_feat_cont(isnan(lay_feat_cont)) = 0; % Assign 0 to NaN values


    %[B, S] = svd(lay_feat_cont,0);
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
    disp('start saving pc')
    save([saveroot,'PredNet_feature_maps_pca_layer',num2str(e_index),'.mat'], 'B', 's', '-v7.3');

    disp('finish saving pc')
end
disp('FINISH PCA')

%% Processed the dimension-reduced 
clc;clear

% layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/E0';'/E1';'/E2';'/E3'}; 
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3'; '/Ahat4';'/E0';'/E1';'/E2';'/E3';'/E4'}; 

% layer type 
layernumber = 'layers5_32c'; 

%128*160 img size
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/06_Cam-CAN/result/modified_prednet/h5/layer5/';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).

srate = 25; 

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);

disp('start dimesion reduction')


% Dimension reduction for CNN features
for lay = 1: length(layername)
    % Dimension reduction for training data
    disp(['Layer: ',layername{lay}]);

    % Determine secpath based on layername
    if contains(layername{lay}, 'Ahat')
        secpath = [dataroot, 'Ahat.h5'];  % If layername contains 'Ahat'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/Ahat/'];

    elseif contains(layername{lay}, 'E')
        secpath = [dataroot, 'E.h5'];  % If layername contains 'E'
        saveroot = ['/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/CC_test/encoding_analysis/modified_prednet/',layernumber,'/E/'];

    else
        disp('Layer name does not match expected pattern.');
        continue;  % Skip if layer name doesn't match
    end

    lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);
    load([saveroot,'PredNet_feature_maps_pca_layer',num2str(lay),'.mat'], 'B');

    if exist(secpath,'file')==2
        lay_feat = h5read(secpath,[layername{lay}]);
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

        dim = size(lay_feat);
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

        % Apply PCA transformation
        Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components

        ts = conv2(hrf,Y); % convolude with hrf
        ts = ts(4*srate+1:4*srate+size(Y,1),:); % Crop the beginning to match the HRF peak
        downsampling_factor = round(srate * 2.47);  % Round to an integer
        ts = ts(1:downsampling_factor:end, :);  % Downsample by the calculated factor
        disp('start saving dimesion reduction for train')

        h5create([saveroot,'PredNet_feature_maps_pcareduced.h5'],[layername{lay},'/data'],...
            [size(ts)],'Datatype','single');
        h5write([saveroot,'PredNet_feature_maps_pcareduced.h5'], [layername{lay},'/data'], ts);
    end
end
disp('finish train ts')


