clc;clear

%% calculate the temporal mean and standard deviation of the feature time series of CNN units
% CNN layer labels
layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'; '/Ahat5'; '/Ahat6';}; 

%128*160 img size
dataroot = '/local_raid3/03_user/jungmin/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_3_32_64_96_192channel/h5/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet_ver1/layers5_3_32_64_96_192channel/Ahat/';

%227*227 img size
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';

disp('start mean')
% calculate the temporal mean 
for lay = 1: length(layername)
    N = 0; 
    for seg = 1 : 8
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath =  [dataroot,'seg', num2str(seg), '_Ahat', '.h5'];

        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay}]);
            dim = size(lay_feat); % convolutional layers: #kernel*fmsize1*fmsize2*#frames
            if seg == 1 
                lay_feat_mean = zeros([dim(1:end-1),1]);
            end
            lay_feat_mean = lay_feat_mean + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_mean = lay_feat_mean/N;
    
    disp('start saving mean')     
    h5create([saveroot,'PredNet_feature_maps_avg.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data'], lay_feat_mean);
end

disp('start std')
% calculate the temporal standard deviation
for lay = 1: length(layername)
    lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    N = 0;
    for seg = 1 : 8
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath =  [dataroot,'seg', num2str(seg), '_Ahat', '.h5'];
        if exist(secpath,'file')
            lay_feat = h5read(secpath,[layername{lay}]);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = lay_feat.^2;
            dim = size(lay_feat);
            if seg == 1 
                lay_feat_std = zeros([dim(1:end-1),1]);
            end
            lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_std = sqrt(lay_feat_std/(N-1));
    lay_feat_std(lay_feat_std==0) = 1;
    
    disp('start saving std')
    h5create([saveroot,'PredNet_feature_maps_std.h5'],[layername{lay},'/data'],...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data'], lay_feat_std);
end

%% Reduce the dimension of the PredNet features for encoding models - %128*160 img size Ahat only works
% Dimension reduction by using principal component analysis (PCA)
% Here provides two ways to compute the PCAs. If the memory is big enough
% to directly calculate the svd, then use the method 1, otherwise use the
% SVD-updating algorithm (Zha and Simon, 1999; Zhao et al., 2006; Wen et al. 2017).

% % % % % % % % % Direct SVD % % % % % % % % % % % % % %
% This requires big memory to compute. It's better to reduce the
% frame rate of videos or use the SVD-updating algorithm.

for lay = 1: length(layername)
    disp('load mean, std')
    lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);
    lay_feat_mean = lay_feat_mean(:);
    lay_feat_std = lay_feat_std(:);

    % Concatenating the feature maps across training movie segments. Ensure that 
    % the memory is big enough to concatenate all movie segments. Otherwise, 
    % use the SVD-updating algorithm. 

    for seg = 1 : 8
        disp(['Layer: ',layername{lay},'; Seg: ', num2str(seg)]);
        secpath =  [dataroot,'seg', num2str(seg), '_Ahat', '.h5'];

        lay_feat = h5read(secpath,[layername{lay}]);
        dim = size(lay_feat);
        Nu = prod(dim(1:end-1)); % number of units
        Nf = dim(end); % number of frames
        lay_feat = reshape(lay_feat,Nu,Nf);% Nu*Nf

%         lay_feat = lay_feat(:,1:3:end); % downsample if encounter memory issue
%         Nf = size(lay_feat,2);

        if seg == 1
            lay_feat_cont = zeros(Nu, Nf*18,'single');
        end
        lay_feat_cont(:, (seg-1 )*Nf+1:seg*Nf) = lay_feat;
    end
    disp('start standardization')
    % standardize the time series for each unit  
    lay_feat_cont = bsxfun(@minus, lay_feat_cont, lay_feat_mean);
    lay_feat_cont = bsxfun(@rdivide, lay_feat_cont, lay_feat_std);
    lay_feat_cont(isnan(lay_feat_cont)) = 0; % assign 0 to nan values

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
    save([saveroot,'PredNet_feature_maps_pca_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');

    disp('finish saving pc')
end
disp('FINISH PCA')
%% %128*160 img size-E, 227*227 img size - Ahat, E 
% % % % % % % % % % % SVD-updating algorithm % % % % % % % % % 
% CNN layer labels
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'; '/Ahat5'; '/Ahat6';}; 
% 
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/original_prednet/layers5_3_32_64_96_192channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/layers5_3_32_64_96_192channel/original_prednet/E/';

%227*227 img size
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';


disp('start svd update')
% % % % % % % % % % % SVD-updating algorithm % % % % % % % % %
Niter = 2; % number of iteration to compute principle component

for lay = 1: length(layername)
    lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);

    k0 = 0;
    percp = 0.99; % explain 99% of the variance of every movie segments
    for iter =  1  : Niter
        for seg = 1 : 18
            disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg),'; Comp:',num2str(k0)]);
            secpath =  [dataroot,'seg', num2str(seg), '_E', '.h5'];
            if exist(secpath,'file')==2
                lay_feat = h5read(secpath,[layername{lay}]);
                lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
                lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
                lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

                dim = size(lay_feat)
                lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
                if (seg == 1) && (iter == 1) 
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
                else
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
                end  
            else
               disp('file doesn''t exist'); 
            end
        end
    end
    s = diag(S);

    disp('start saving svd update')

    % save principal components
    save([saveroot,'PredNet_feature_maps_svd_layer',num2str(lay),'.mat'], 'B', 's', '-v7.3');
end
disp('FINISH PCA_updating SVD version')

%% Processed the dimension-reduced 
clc;clear

layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'; '/Ahat5'; '/Ahat6';}; 

% 
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/original_prednet/layers5_3_32_64_96_192channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/layers5_3_32_64_96_192channel/original_prednet/E/';

%227*227 img size
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 30; %zhongmin liu

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);

disp('start dimesion reduction')


% Dimension reduction for CNN features
for lay = 1: length(layername)
    lay_feat_mean = h5read([saveroot,'PredNet_feature_maps_avg.h5'], [layername{lay},'/data']);
    lay_feat_std = h5read([saveroot,'PredNet_feature_maps_std.h5'], [layername{lay},'/data']);
    load([saveroot,'PredNet_feature_maps_pca_layer',num2str(lay),'.mat'], 'B');
    % load([saveroot,'PredNet_feature_maps_svd_layer',num2str(lay),'.mat'], 'B');

    % Dimension reduction for training data
    for seg = 1 : 18
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath =  [dataroot,'seg', num2str(seg), '_Ahat', '.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay}]);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(4*srate+1:4*srate+size(Y,1),:);
            ts = ts(srate+1:2*srate:end,:); % downsampling to match fMRI

            disp('start saving dimesion reduction for train')

            h5create([saveroot,'PredNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(ts)],'Datatype','single');
            h5write([saveroot,'PredNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
        end
    end   
    disp('finish train ts')
    % Dimension reduction for testing data
    for seg = 1 : 5
        disp(['Layer: ', num2str(lay),'; Test: ',num2str(seg)]);
        secpath =  [dataroot,'seg', num2str(seg), '_Ahat', '.h5'];
        if exist(secpath,'file')==2
            lay_feat = h5read(secpath,[layername{lay}]);
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
            
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            Y = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
            
            ts = conv2(hrf,Y); % convolude with hrf
            ts = ts(4*srate+1:4*srate+size(Y,1),:);
            ts = ts(srate+1:2*srate:end,:); % downsampling

            disp('start saving dimesion reduction for test')

            h5create([saveroot,'PredNet_feature_maps_pcareduced_test', num2str(seg),'.h5'],[layername{lay},'/data'],...
                [size(ts)],'Datatype','single');
            h5write([saveroot,'PredNet_feature_maps_pcareduced_test', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
        end
    end  
    disp('finish test ts')
    
end

%% Concatenate the dimension-reduced CNN features of training movies
% CNN layer labels
layername = {'/Ahat0';'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4'; '/Ahat5'; '/Ahat6';}; 
% 
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/original_prednet/layers5_3_32_64_96_192channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/layers5_3_32_64_96_192channel/original_prednet/E/';


%227*227 img size
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/05_AlexNet/result/modified_prednet/layers5_96_256_384_384_256channel/h5/';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/modified_prednet/layers5_96_256_384_384_256channel/Ahat/';

disp('start concatenated')

for lay = 1: length(layername)
    for seg = 1 : 18
        disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
        secpath = [saveroot,'PredNet_feature_maps_pcareduced_seg', num2str(seg),'.h5'];     
        lay_feat = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
        dim = size(lay_feat);
        Nf = dim(1); % number of frames
        if seg == 1
           Y = zeros([Nf*18, dim(2)],'single'); 
        end
        Y((seg-1)*Nf+1:seg*Nf, :) = lay_feat;
    end
    h5create([saveroot,'PredNet_feature_maps_pcareduced_concatenated.h5'],[layername{lay},'/data'],...
        [size(Y)],'Datatype','single');
    h5write([saveroot,'PredNet_feature_maps_pcareduced_concatenated.h5'], [layername{lay},'/data'], Y);
end

%% jmlee: concat for testing movie as well 
% % CNN layer labels
% layername = {'/Ahat1';'/Ahat2';'/Ahat3';'/Ahat4';}; 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/AlexNet_test/encoding_analysis/';
% 
% for lay = 1: length(layername)
%     for seg = 1 : 5
%         disp(['Layer: ', num2str(lay),'; Seg: ',num2str(seg)]);
%         secpath = [saveroot,'PredNet_feature_maps_pcareduced_test', num2str(seg),'.h5'];     
%         lay_feat = h5read(secpath,[layername{lay},'/data']);% #time-by-#components
%         dim = size(lay_feat);
%         Nf = dim(1); % number of frames
%         if seg == 1
%            Y = zeros([Nf*5, dim(2)],'single'); 
%         end
%         Y((seg-1)*Nf+1:seg*Nf, :) = lay_feat;
%     end
%     h5create([saveroot,layername{lay}(2),'/','PredNet_feature_maps_pcareduced_concatenated_test.h5'],[layername{lay},'/data'],...
%         [size(Y)],'Datatype','single');
%     h5write([saveroot,layername{lay}(2),'/','PredNet_feature_maps_pcareduced_concatenated_test.h5'], [layername{lay},'/data'], Y);
% end
% 
