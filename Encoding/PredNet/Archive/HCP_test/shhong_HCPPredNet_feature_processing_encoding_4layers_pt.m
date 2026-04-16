%% Step2: Concate X
clc; clear;

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/prediction_time_delay';
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers_32channel/h5';

features = {'Ahat', 'E'};
time_delay = {1, 2, 3};

for i = 1:3
    disp(['layer is ', num2str(i)])

    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)
        
        total_lay_feat = [];

        for seg = 1:4
        
            % calculate the temporal mean 
            secpath = [dataroot, '/', sprintf('MOVIE%d_%s%d.h5', seg, feature, i)];
            lay_feat = h5read(secpath,'/predimg');

            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

            for j = length(time_delay)

               td = time_delay{j};
               concat_lay_feat = vertcat(lay_feat(1:end-25*td,:), lay_feat(25*td+1:end,:)); 

               h5_name = sprintf([saveroot, '/X/MOVIE%d_%s%d_time_delayed.h5'], seg, feature, i);
               h5create(h5_name, '/data', size(concat_lay_feat));
               h5write(h5_name, '/data', concat_lay_feat);

            end
        end
    end
end

%% Step1: calculate the temporal mean and standard deviation of the feature time series of CNN units
clc;clear
% dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/prediction_time_delay';

features = {'E'}; % 'Ahat', 

for i = 1:3
    
    X={};

    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        % calculate the temporal mean 
        secpath1 = [saveroot, '/', 'X', '/', sprintf('MOVIE1_%s%d_time_delayed.h5', feature, i)];
        lay_feat1 = h5read(secpath1,'/data');
        secpath2 = [saveroot, '/', 'X', '/', sprintf('MOVIE2_%s%d_time_delayed.h5', feature, i)];
        lay_feat2 = h5read(secpath2,'/data');
        secpath3 = [saveroot, '/', 'X', '/', sprintf('MOVIE3_%s%d_time_delayed.h5', feature, i)];
        lay_feat3 = h5read(secpath3,'/data');
        secpath4 = [saveroot, '/', 'X', '/', sprintf('MOVIE4_%s%d_time_delayed.h5', feature, i)];
        lay_feat4 = h5read(secpath4,'/data');
        
        lay_feat = horzcat(lay_feat1, lay_feat2, lay_feat3, lay_feat4);
        dim = size(lay_feat)
    
        lay_feat_mean = sum(lay_feat, length(dim)) / dim(end);
        
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = lay_feat.^2;
        dim = size(lay_feat);
    
        lay_feat_std = zeros([dim(1:end-1),1]); 
        lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
    
        lay_feat_std = sqrt(lay_feat_std/(dim(end)-1));
        lay_feat_std(lay_feat_std==0) = 1;
                
        disp(['save average for layer', num2str(i)])
        
        h5create([saveroot, '/', 'X', '/', sprintf('%s%d_time_delayed_concat_feature_maps.h5', feature, i)],'/data',...
            [size(lay_feat)],'Datatype','single');
        h5write([saveroot, '/', 'X', '/', sprintf('%s%d_time_delayed_concat_feature_maps.h5', feature, i)],'/data', lay_feat);
        
        h5create([saveroot, '/', 'X', '/', sprintf('%s%d_time_delayed_feature_maps_avg.h5', feature, i)],'/data',...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([saveroot, '/', 'X', '/', sprintf('%s%d_time_delayed_feature_maps_avg.h5', feature, i)],'/data', lay_feat_mean);

        h5create([saveroot, '/', 'X', '/', sprintf('%s%d_time_delayed_feature_maps_std.h5', feature, i)],'/data',...
            [size(lay_feat_std)],'Datatype','single');
        h5write([saveroot,'/','X', '/',  sprintf('%s%d_time_delayed_feature_maps_std.h5', feature, i)],'/data',lay_feat_std);        
        
    end
end

%% Step2: PCA   

clc;clear

%dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test/prediction_time_delay';

% % % % % % % % % % SVD-updating algorithm % % % % % % % % %

Niter = 2; % number of iteration to compute principle component
percp = 0.90; % explain 99% of the variance of every movie segments
disp(['percp is ', num2str(percp)])

features = { 'Ahat', 'E'};

for i = 1:3
    disp(['layer is ', num2str(i)])

    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature);
     

        k0 = 0;
        for iter =  1  : Niter
            disp(['iteration #: ', num2str(iter)])
            
            %load lay_feat 
            disp('load regularized lay_feat')
            
            secpath = [saveroot,'/','feature_maps', '/', sprintf('%s%d_1-4_concatenated_feature.h5', feature, i)];
            lay_feat = h5read(secpath,'/data');
            dim = size(lay_feat)

            disp('amri_sig_isvd start')
        
            if iter == 1
                [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
            else
                [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
            end  
        
            disp(['Iteration ', num2str(iter), ' completed for', feature, num2str(i)]);
        end
        
        disp('diag(s) begins')
        
        s = diag(S);
        
        disp('start saving SVD_layer');
        % save principal components
        save([saveroot, '/', 'X', '/', 'PCA', '/', sprintf('%s%d_feature_maps_svd_0.90.mat', feature, i)], 'B', 's', '-v7.3');
        disp(['SVD_layer for', feature, ' saved']);
        
        clear B s S lay_feat
    end
end
