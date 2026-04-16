%% Process the AlexNet features for bivariate analysis to relate CNN units to brain voxels
clc;clear

% CNN layer labels
layername = {'/A0.npy'; '/A1';'/A2';'/A3';'/A4';}; 

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 30; 

% Here is an example of using predefined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);

%128*160 img size
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/original_prednet/layers5_3_32_64_96_192channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/layers5_3_32_64_96_192channel/original_prednet/E/';

%227*227 img size
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/bivariate_analysis/modified_prednet/layers5_96_256_384_384_256channel/A/';

clc;

%% Training movies
for seg = 1 : 18
    secpath = [dataroot,'seg', num2str(seg), '_A', '.h5']
    
    for lay = 1 : length(layername)
        disp(['Seg: ', num2str(seg), '; Layer: ',layername{lay}]);
        % info = h5info(secpath);
        lay_feat = h5read(secpath,[layername{lay}]);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        if lay < length(layername)
            lay_feat = log10(lay_feat + 0.01); % log-transformation except the last layer
        end
        ts = conv2(hrf,lay_feat'); % convolude with hrf
        ts = ts(4*srate+1:4*srate+Nf,:);
        ts = ts(srate+1:2*srate:end,:)'; % downsampling
        ts = reshape(ts,[dim(1:end-1),240]);
        
        % check time series
        % figure;plot(squeeze(ts(25,25,56,:)));

        h5create([saveroot,'PredNet_feature_maps_processed_seg', num2str(seg), '_A', '.h5'],[layername{lay},'/data'],...
            [size(ts)],'Datatype','single');
        h5write([saveroot,'PredNet_feature_maps_processed_seg', num2str(seg), '_A', '.h5'], [layername{lay},'/data'], ts);
    end
end

% Testing movies
for test = 1 : 5
    secpath = [dataroot,'test', num2str(test), '_A', '.h5']
    for lay = 1 : length(layername)
        disp(['Test: ', num2str(test), '; Layer: ',layername{lay}]);
        lay_feat = h5read(secpath,[layername{lay}]);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf
        if lay < length(layername)
            lay_feat = log10(lay_feat + 0.01); % log-transformation
        end
        ts = conv2(hrf,lay_feat'); % convolude with hrf
        ts = ts(4*srate+1:4*srate+Nf,:);
        ts = ts(srate+1:2*srate:end,:)'; % downsampling
        ts = reshape(ts,[dim(1:end-1),240]);

        h5create([saveroot,'PredNet_feature_maps_processed_test', num2str(test), '_A', '.h5'],[layername{lay},'/data'],...
            [size(ts)],'Datatype','single');
        h5write([saveroot,'PredNet_feature_maps_processed_test', num2str(test), '_A', '.h5'], [layername{lay},'/data'], ts);
    end
end