

%% Step1: calculate the temporal mean and standard deviation of the feature time series of CNN units

clc;clear

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';


%make folder
folder_name = [saveroot,'/', 'feature_maps','/', '6layers_32channel', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

Prediction_features = {'run1_Ahat', 'run2_Ahat', 'run3_Ahat', 'run4_Ahat', 'run5_Ahat', 'run6_Ahat', 'run7_Ahat', 'run8_Ahat'}
Error_features={'run1_E', 'run2_E', 'run3_E', 'run4_E', 'run5_E', 'run6_E', 'run7_E', 'run8_E'};

% calculate the Prediction feature temporal mean
for lay = 0 : 5
    N = 0;
    for seg = 1 : 8
        feature = Prediction_features{seg};
        disp(feature)

        name_index = strfind(feature, '_');  % Find the index of the underscore
        feature_name= feature(name_index+1:end);  % Extract the part after the underscore
        disp(feature_name)

        secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
        if exist(secpath,'file')
            lay_feat = h5read(secpath,'/predimg');
            dim = size(lay_feat)
            if seg == 1
                lay_feat_mean = zeros([dim(1:end-1),1]);
            end
            lay_feat_mean = lay_feat_mean + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_mean = lay_feat_mean/N;

    h5create([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature_name, lay)],'/data',...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature_name, lay)],'/data', lay_feat_mean);
end

% calculate the Prediction temporal standard deviation
for lay = 0 : 5
    lay_feat_mean = h5read([folder_name,sprintf('Ahat%d_feature_maps_avg.h5', lay)], '/data');
    N = 0;
    for seg = 1 : 8
        feature = Prediction_features{seg};
        disp(feature)

        name_index = strfind(feature, '_');  % Find the index of the underscore
        feature_name= feature(name_index+1:end);  % Extract the part after the underscore
        disp(feature_name)

        secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
        if exist(secpath,'file')
            lay_feat = h5read(secpath,'/predimg');
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
    h5create([folder_name, sprintf('%s%d_feature_maps_std.h5', feature_name, lay)],'/data',...
        [size(lay_feat_std)],'Datatype','single');
    h5write([folder_name, sprintf('%s%d_feature_maps_std.h5', feature_name, lay)],'/data', lay_feat_std);
end


%clear memory
clear lay_feat lay_feat_std lay_feat_mean

% calculate the Error feature temporal mean
for lay = 0 : 5
    N = 0;
    for seg = 1 : 8
        feature = Error_features{seg};
        disp(feature)

        name_index = strfind(feature, '_');  % Find the index of the underscore
        feature_name= feature(name_index+1:end);  % Extract the part after the underscore
        disp(feature_name)

        secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
        if exist(secpath,'file')
            lay_feat = h5read(secpath,'/predimg');
            dim = size(lay_feat)
            if seg == 1
                lay_feat_mean = zeros([dim(1:end-1),1]);
            end
            lay_feat_mean = lay_feat_mean + sum(lay_feat,length(dim));
            N = N  + dim(end);
        end
    end
    lay_feat_mean = lay_feat_mean/N;

    h5create([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature_name, lay)],'/data',...
        [size(lay_feat_mean)],'Datatype','single');
    h5write([folder_name, sprintf('%s%d_feature_maps_avg.h5', feature_name, lay)],'/data', lay_feat_mean);
end

% calculate the Error temporal standard deviation
for lay = 0 : 5
    lay_feat_mean = h5read([folder_name,sprintf('E%d_feature_maps_avg.h5', lay)], '/data');
    N = 0;
    for seg = 1 : 8
        feature = Error_features{seg};
        disp(feature)

        name_index = strfind(feature, '_');  % Find the index of the underscore
        feature_name= feature(name_index+1:end);  % Extract the part after the underscore
        disp(feature_name)

        secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
        if exist(secpath,'file')
            lay_feat = h5read(secpath,'/predimg');
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
    h5create([folder_name, sprintf('%s%d_feature_maps_std.h5', feature_name, lay)],'/data',...
        [size(lay_feat_std)],'Datatype','single');
    h5write([folder_name, sprintf('%s%d_feature_maps_std.h5', feature_name, lay)],'/data', lay_feat_std);
end

%% PCA - each segment PCA concatenation 
clc; clear

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

%make folder
folder_name = [saveroot,'/', 'X','/', '6layers_32channel', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

Prediction_features = {'run1_Ahat', 'run2_Ahat', 'run3_Ahat', 'run4_Ahat', 'run5_Ahat', 'run6_Ahat', 'run7_Ahat', 'run8_Ahat'}
Error_features={'run1_E', 'run2_E', 'run3_E', 'run4_E', 'run5_E', 'run6_E', 'run7_E', 'run8_E'};

Niter = 2; % number of iteration to compute principle component

%prediction
disp('START PREDICTION LAYER');
for lay = 0 : 5
    k0 = 0;
    percp = 0.90; % explain 90% of the variance of every movie segments
    lay_feat_mean = h5read(fullfile(saveroot,'/', 'feature_maps','/', '6layers_32channel', '/',sprintf('Ahat%d_feature_maps_avg.h5',  lay)), '/data');
    disp(sprintf('%s%d_feature_maps_avg.h5', feature, lay))
    lay_feat_std = h5read(fullfile(saveroot,'/', 'feature_maps','/', '6layers_32channel', '/', sprintf('Ahat%d_feature_maps_std.h5',  lay)), '/data');
    disp(sprintf('%s%d_feature_maps_std.h5', feature, lay))
    for iter = 1:Niter
        for feature_idx = 1:numel(Prediction_features)
            feature = Prediction_features{feature_idx};
            disp(sprintf('feature_idx: %d', feature_idx))
            feature_name = feature(strfind(feature, '_')+1:end);

            if strcmp(feature_name, 'Ahat')
                disp(['Layer: ', num2str(lay),'; Seg: ',feature,'; Comp:',num2str(k0)]);

                % Load lay_feat
                disp('load lay_feat')
                secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
                lay_feat = h5read(secpath,'/predimg');

                % standardize of lay_feat
                disp('normalization begins')
                lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
                lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
                lay_feat(isnan(lay_feat)) = 0; % Assign 0 to NaN values

                dim = size(lay_feat)
                lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));  %featurextime
                
                %svd
                if (feature_idx == 1) && (iter == 1)
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
                else
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
                end
            end
        end
    end
    s = diag(S);
    % save principal components
    save([folder_name, sprintf('%s%d_feature_maps_svd_0.90.mat', feature_name, lay)], 'B', 's', '-v7.3');

    % save up memory
    disp(['start remove B s S for memory for layer ',num2str(lay)])
    clear B s S

    % start next layer
    disp('START NEXT LAYER');
end 
clear lay_feat lay_feat_mean lay_feat_std 

%error
disp('START ERROR FEATURE');

for lay = 0 : 5
    k0 = 0;
    percp = 0.90; % explain 90% of the variance of every movie segments
    lay_feat_mean = h5read(fullfile(saveroot,'/', 'feature_maps','/', '6layers_32channel', '/',sprintf('E%d_feature_maps_avg.h5',  lay)), '/data');
    disp(sprintf('%s%d_feature_maps_avg.h5', feature, lay))
    lay_feat_std = h5read(fullfile(saveroot,'/', 'feature_maps','/', '6layers_32channel', '/', sprintf('E%d_feature_maps_std.h5',  lay)), '/data');
    disp(sprintf('%s%d_feature_maps_std.h5', feature, lay))
    for iter = 1:Niter
        for feature_idx = 1:numel(Error_features)
            feature = Error_features{feature_idx};
            feature_name = feature(strfind(feature, '_')+1:end)

            if strcmp(feature_name, 'E')            
                disp(['Layer: ', num2str(lay),'; Seg: ',feature,'; Comp:',num2str(k0)]);
         
                % Load lay_feat
                secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
                lay_feat = h5read(secpath,'/predimg');

                % standardize of lay_feat
                disp('normalization begins')
                lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
                lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
                lay_feat(isnan(lay_feat)) = 0; % Assign 0 to NaN values

                dim = size(lay_feat)
                lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));  %featurextime
                
                %svd
                if (feature_idx == 1) && (iter == 1)
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
                else
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
                end
            end
        end
    end
    s = diag(S);
    % save principal components
    save([folder_name, sprintf('%s%d_feature_maps_svd_0.90.mat', feature_name, lay)], 'B', 's', '-v7.3');
 
    % save up memory
    disp(['start remove B s S for memory for layer ',num2str(lay)])
    clear B s S

   % start next layer
    disp('START NEXT LAYER');
end

%% Step3: Processed the hrf 
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

srate = 25; %original code is at TR=2, [if TR=1 && srate is 25] then make it into 12.5 

p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);

Prediction_features = {'run1_Ahat', 'run2_Ahat', 'run3_Ahat', 'run4_Ahat', 'run5_Ahat', 'run6_Ahat', 'run7_Ahat', 'run8_Ahat'}
Error_features={'run1_E', 'run2_E', 'run3_E', 'run4_E', 'run5_E', 'run6_E', 'run7_E', 'run8_E'};

svd_folder = [saveroot,'/', 'X','/', '6layers_32channel', '/'];

%Prediction
disp('START PREDICTION FEATURE');
for lay=0
    disp(num2str(lay))

    lay_feat_mean = h5read([saveroot, '/', 'feature_maps','/', '6layers_32channel', '/',sprintf('Ahat%d_feature_maps_avg.h5', lay)], '/data');
    disp(sprintf('Ahat%d_feature_maps_avg.h5', lay))

    lay_feat_std = h5read([saveroot, '/', 'feature_maps','/', '6layers_32channel', '/', sprintf('Ahat%d_feature_maps_std.h5', lay)], '/data');
    disp(sprintf('Ahat%d_feature_maps_std.h5', lay))

    load([svd_folder, sprintf('Ahat%d_feature_maps_svd_0.90.mat', lay)], 'B');
    disp(sprintf('Ahat%d_feature_maps_svd_0.90.mat', lay))


    for seg = 1 : 8
        feature = Prediction_features{seg}
        feature_name = feature(strfind(feature, '_')+1:end)

        %load lay_feat
        disp('load lay_feat')
        secpath = [dataroot, '/', 'h5', '/', sprintf('%s%d.h5', feature, lay)]
        lay_feat = h5read(secpath,'/predimg');

        disp('compute lay_feat regulaization')
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

        disp('reshape lay_feat')
        dim = size(lay_feat)
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

        disp('compute X with svd applied')
        X = lay_feat'*B/sqrt(size(B,1)); % X: #time-by-#components

        disp('compute ts')
        ts = conv2(hrf,X); % convolude with hrf
        ts = ts(4*srate+1:4*srate+size(X,1),:);
        ts = ts(srate+1:2*srate:end,:); % downsampling to match fMRI

        disp('start saving')

        filename = [sprintf('%s%d_feature_maps_svd_0.90_hrf_seg%s.mat', feature_name,lay, num2str(seg))]
        save([svd_folder, filename], 'ts', '-v7.3');

        clear ts lay_feat
    end
end

%Error
disp('START ERROR FEATURE');

for lay=0:5
    disp(num2str(lay))

    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/','6layers_32channel', '/',  sprintf('E%d_feature_maps_avg.h5',lay)], '/data');
    disp(sprintf('E%d_feature_maps_avg.h5', lay))

    lay_feat_std = h5read([saveroot, '/','feature_maps', '/','6layers_32channel', '/', sprintf('E%d_feature_maps_std.h5',  lay)], '/data');
    disp(sprintf('E%d_feature_maps_std.h5', lay))

    load([svd_folder, sprintf('E%d_feature_maps_svd_0.90.mat', lay) ], 'B');
    disp(sprintf('E%d_feature_maps_svd_0.90.mat', lay))

    for seg = 1 : 8
        feature = Error_features{seg}
        feature_name = feature(strfind(feature, '_')+1:end)

        %load lay_feat
        disp('load lay_feat')
        secpath = [dataroot, '/', 'h5', '/',  sprintf('%s%d.h5', feature,lay)];
        lay_feat = h5read(secpath,'/predimg');

        disp('compute lay_feat regulaization')
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values

        disp('reshape lay_feat')
        dim = size(lay_feat);
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));

        disp('compute X with svd applied')
        X = lay_feat'*B/sqrt(size(B,1)); % X: #time-by-#components

        disp('compute ts')
        ts = conv2(hrf,X); % convolude with hrf
        ts = ts(4*srate+1:4*srate+size(X,1),:);
        ts = ts(srate+1:2*srate:end,:); % downsampling to match fMRI

        disp('start saving')

        filename = [sprintf('%s%d_feature_maps_svd_0.90_hrf_seg%s.mat', feature_name,lay, num2str(seg))];
        save([svd_folder, filename], 'ts', '-v7.3');

        clear ts lay_feat
    end
end

%% W
clc; clear
disp('Load model featuremaps')

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

features = { 'Ahat', 'E'};
svd_folder = [saveroot,'/', 'X','/', '6layers_32channel', '/'];

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:5
        for seg = 1:8
            filename = sprintf('%s/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', svd_folder, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end

end

%Voxelwise_encoding 
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper 

%load Y
% load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']); % 10k
load([fmriroot, '/' , 'Y', '/', 'fMRI_59k_cortex.mat']); % 59k
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 4:15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:8
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:5
                disp(['Layer: ', num2str(layer)]);
                 % Y calculate 
                switch seg
                    case 1
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 2
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 3
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{4}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 4
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{5},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 5
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{6},sub_cii{i}{7}, sub_cii{i}{8});
                    case 6
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{7}, sub_cii{i}{8});
                    case 7
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{6}, sub_cii{i}{8});
                    case 8
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4},sub_cii{i}{5},sub_cii{i}{6}, sub_cii{i}{7});
                end
                 
                % X calculate
                X_tot = {};
                for s = 1:8
                    if s ~= seg
                        disp(['s is ', num2str(s)])
                        disp([sprintf('%s%d_seg%d', feature, layer, s)])
                        X_tot{end+1} = concatenated.(sprintf('%s%d_seg%d', feature, layer, s));
                    end
                end
                X = cat(1, X_tot{:});
                
                %voxelwise encoding
                [W, Rmat, Lambda] = voxelwise_encoding(X, Y', lambda, nfold);
                
                % save W (feature x voxel)
                folder_name = [saveroot,'/', 'W','/','6layers_32channel','/','59k', '/', sprintf('seg%d',seg), '/']
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end
                
                name = [num2str(subject), '_', feature, num2str(layer), '_W_svd0.9_s3.mat']
                save([folder_name, name], 'W', '-v7.3');
            end
        end
    end
end

%% Step5: Yhat
clc; clear
disp('Load model featuremaps')

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = {'Ahat','E'};

svd_folder = [saveroot,'/', 'X','/', '6layers_32channel', '/'];

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:5
        for seg = 1:8
            filename = sprintf('%s/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', svd_folder, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end
end

%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_59k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:8
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:5
                disp(['Layer: ', num2str(layer)]);

                % X calculate
                X = {};
                for s = 1:8
                    if s == seg
                        disp(['s is ', num2str(s)])
                        disp([sprintf('%s%d_seg%d', feature, layer, s)])
                        X = concatenated.(sprintf('%s%d_seg%d', feature, layer, s));
                    end
                end

                W = load([saveroot,'/', 'W', '/', '6layers_32channel','/', '59k', '/', sprintf('seg%d',seg), '/', num2str(subject), '_', feature, num2str(layer), '_W_svd0.9_s3.mat']);
                name = ['Y_pred_', feature, num2str(layer)]; 
                eval([name, '= X*W.W;']);
                
                folder_name = [saveroot,'/', 'Y_hat','/','6layers_32channel','/', '59k', '/', sprintf('seg%d',seg), '/']
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end
                
                %save Y_hat
                Yhat = eval(name);
                name2 = strcat(num2str(subject),'_', feature , num2str(layer), '_pred_svd0.9_s3.mat');
               
                save([folder_name,  name2], 'Yhat', '-v7.3');
            end
        end
    end
end


%% step6: correaltion between Y_predict and actual Y for test - single segment
clc; clear
disp('correaltion and p_values compute start')

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);


for i = 1:15
    subject = num_subjects{i};

    disp(['correalting' , subject])

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature)

        for layer= 0:5
            for seg = 1:8
                folder_name = [saveroot,'/', 'correlations','/', '6layers_32channel','/', sprintf('seg%d',seg)]
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end

                Y_pred = load([saveroot,'/', 'Y_hat','/','6layers_32channel', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pred_svd0.9_s3', '.mat')]);
                Y_pred = Y_pred.Yhat;

                disp(['seg is ' , num2str(seg)])

                if seg == 1
                    Y = sub_cii{i}{1};
                elseif seg == 2
                    Y = sub_cii{i}{2};
                elseif seg == 3
                    Y = sub_cii{i}{3};
                elseif seg == 4
                    Y = sub_cii{i}{4};
                elseif seg == 5
                    Y = sub_cii{i}{5};
                elseif seg == 6
                    Y = sub_cii{i}{6};
                elseif seg == 7
                    Y = sub_cii{i}{7};
                elseif seg == 8
                    Y = sub_cii{i}{8};
                end

                [correlations, p_values] = auto_corr(Y',Y_pred);

                correlations = correlations';

                % save correlation (feature x voxel)
                disp('save correlations')

                name = strcat(num2str(subject),'_', feature , num2str(layer), '_corr_svd0.9_S3.mat')
                save([saveroot,'/', 'correlations','/', '6layers_32channel','/10k/',sprintf('seg%d',seg), '/', name], 'correlations', '-v7.3');

                name_pvalues = strcat(num2str(subject),'_', feature , num2str(layer), '_pval_svd0.9_S3.mat')
                save([saveroot,'/', 'correlations','/', '6layers_32channel','/10k/',sprintf('seg%d',seg), '/', name_pvalues], 'p_values', '-v7.3');
                
            end
        end
    end
end

%% Step 6: concat Yhat for all segments + correalting with total movie fMRI
clc; clear

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

folder_name = [saveroot,'/', 'correlations','/', '6layers_32channel','/', '10k', '/', 'total', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

for i = 1:15
    subject = num_subjects{i};
    disp(i)
    disp(['Voxelwise_encoding start sub ', num2str(subject)])    

    for feature_idx = 2:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer=0:5
            for seg = 1:8
                Y_hat = load([saveroot,'/', 'Y_hat','/','6layers_32channel', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pred_svd0.9_s3', '.mat')]);
                disp([saveroot,'/', 'Y_hat','/','6layers_32channel', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer), '_pred_svd0.9_S3.mat')])

                %zscore
                X{seg} = zscore(Y_hat.Yhat);

            end
            Y_pred = sprintf('%s%d_seg1_8', feature, layer)

            %concatenated zscored Y_hat for all seg
            concatenated.(Y_pred)= cat(1, X{:});

            %correaltions
            %load Y 
            sub_cii1_8 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
            [correlations, p_values] = auto_corr(sub_cii1_8',concatenated.(Y_pred));

            correlations = correlations';

            % save correlation (feature x voxel)
            disp('save correlations')

            name = strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3', '.mat')
            save([folder_name, name], 'correlations', '-v7.3');

            name_pval = strcat(num2str(subject),'_', feature , num2str(layer),'_pval_svd0.90_s3', '.mat')
            save([folder_name, name_pval], 'p_values', '-v7.3');
        end
    end
end


%% Step7: make dtseries - total
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

folder_name = [saveroot,'/', 'dt_result','/','6layers_32channel','/', 'total' , '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end
X={} ;

for i = 1:15
    subject = num_subjects{i};
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer=0:5
            % Load cifti -10k
            folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
            files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

            file_path = fullfile(folder_path, files.name);
            cii = ciftiopen(file_path, 'wb_command');
            cii = rmfield(cii, 'cdata');

            %load correaltion
            load([saveroot,'/', 'correlations','/','6layers_32channel','/', 'total', '/', num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3']);
            disp([saveroot,'/', 'correlations','/','6layers_32channel','/', 'total', '/', num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3']);

            % Update cdata and diminfo
            cii.cdata = correlations;
            cii.diminfo{1, 1}.length = size(correlations, 1);
            cii.diminfo{1, 2}.length = size(correlations, 2);
            cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal

            % Concatenate cdata for each layer
            if isempty(X)
                X = cii;
            else
                X.cdata = cat(2, X.cdata, cii.cdata);
            end
        end

        % Save as dtseries
        save_file_name = ([num2str(subject),'_', feature, '.dtseries.nii']);
        ciftisavereset(X, fullfile(folder_name, save_file_name), 'wb_command');

        X={} ;
    end
end

%% Step7: make dtseries - single sub
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};
X={};

cii_10k_LR = ciftiopen('/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.10k_fs_LR.dlabel.nii', 'wb_command');

vertex=cii_10k_LR.cdata;

v1 = find(vertex == 1 | vertex == 181);
v2 = find(vertex == 4 | vertex == 184 );
v3 = find(vertex == 13 | vertex == 19 | vertex == 158 | vertex == 5 | vertex == 193 | vertex == 199 | vertex == 338 | vertex == 185);
v4 = find(vertex == 6 | vertex == 156 | vertex == 186 | vertex == 336);

for i = 1:15
    subject = num_subjects{i};

    for seg = 1:8
        folder_name = [saveroot,'/', 'dt_result','/', '6layers_32channel','/', sprintf('seg%d',seg),'/']
        if not(exist(folder_name,'dir'))
            mkdir(folder_name)
        end
        for feature_idx = 1:numel(features)
            feature = features{feature_idx}
            disp(feature);

            for layer= 0:5

                % Load cifti -10k
                folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
                files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

                file_path = fullfile(folder_path, files.name);
                cii = ciftiopen(file_path, 'wb_command');
                cii = rmfield(cii, 'cdata');


%                 % Load cifti -59k
%                 folder_path = '/combinelab/03_user/jungmin/02_data/03_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results/ses-movie_task-movie_run-1';
%                 files = dir(fullfile(folder_path, 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'));
% 
%                 file_path = fullfile(folder_path, files.name);
%                 cii = ciftiopen(file_path, 'wb_command');
%                 cii = rmfield(cii, 'cdata');

%                 %load correaltion
%                 load([saveroot,'/', 'correlations','/', '6layers_32channel','/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.9_S3.mat')]);
%                 disp([sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3.mat')]);
% 
%                 % Update cdata and diminfo
%                 cii.cdata = correlations;
%                 cii.diminfo{1, 1}.length = size(correlations, 1);
%                 cii.diminfo{1, 2}.length = size(correlations, 2);
%                 cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
%                 
                %load p value
                load([saveroot,'/', 'correlations','/', '6layers_32channel','/10k/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pval_svd0.9_S3.mat')]);
                disp([sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pval_svd0.90_s3.mat')]);

                roi_pval = zeros(size(p_values)).';
                roi_pval(v1) = p_values(v1);
                roi_pval(v2) = p_values(v2);
                roi_pval(v3) = p_values(v3);
                roi_pval(v4) = p_values(v4);

                % Update cdata and diminfo
                cii.cdata = roi_pval;
                cii.diminfo{1, 1}.length = size(p_values.', 1);
                cii.diminfo{1, 2}.length = size(p_values.', 2);
                cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
                
                % Concatenate cdata for each layer
                if isempty(X)
                    X = cii;
                else
                    X.cdata = cat(2, X.cdata, cii.cdata);
                end
            end
            % Save as dtseries
            save_file_name = ([num2str(subject),'_', feature, '_roi_pval.dtseries.nii']);
            ciftisavereset(X, fullfile(folder_name,save_file_name), 'wb_command');

            %initialize
            X={};
        end
    end
end

%% CORR avg .mat 
clc;clear

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

Y_hat = {};


for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:8
        folder_name = [saveroot,'/', 'dt_result','/', '6layers_32channel','/', '10k', '/', 'total','/']
        % if not(exist(folder_name,'dir'))
        %     mkdir(folder_name)
        % end 
        for layer= 1:5
            sum_values = 0;
            for i = 1:15
                subject = num_subjects{i}

                % Load correlation
                load([saveroot,'/', 'correlations','/', '6layers_32channel', '/', '10k', '/', 'total', '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3.mat')]);
                disp([sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3.mat')]);

                sum_values = sum_values + correlations';
            end
        Y_hat{layer+1} = sum_values / 15;
        end
        save([saveroot, '/', 'correlations', '/', '6layers_32channel', '/', '10k', '/', 'total', '/', sprintf('%s_avg_corr_svd0.90_s3.mat', feature)], 'Y_hat', '-v7.3');

        %initalize 
        Y_hat = {};
    end
end 


%% PVALUE avg .mat
clc;clear

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

Y_hat = {};


for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:8
        % folder_name = [saveroot,'/','6layers_32channel','/',  '10k', '/', 'total','/']
%         if not(exist(folder_name,'dir'))
%             mkdir(folder_name)
%         end 
        for layer= 0:5
            sum_values = 0;
            for i = 1:15
                subject = num_subjects{i}

                % Load pvalues
                load([saveroot,'/', 'correlations','/', '6layers_32channel', '/', '10k', '/', 'total', '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pval_svd0.90_s3.mat')]);

                sum_values = sum_values + p_values;
            end
        p_value{layer+1} = sum_values / 15;
        end
        save([saveroot, '/', 'correlations', '/', '6layers_32channel','/','10k', '/', 'total', '/', sprintf('%s_avg_pval_svd0.90_s3.mat', feature)], 'p_value', '-v7.3');

        %initalize 
        p_value = {};
    end
end 


%% make avg dtseries 
clc;clear

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 


% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

X={};

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:8
        folder_name = [saveroot,'/', 'dt_result','/', '6layers_32channel','/', '10k', '/', 'total', '/']
        if not(exist(folder_name,'dir'))
            mkdir(folder_name)
        end 
            for layer=0:5

                % Load cifti -10k
                folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
                files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

                file_path = fullfile(folder_path, files.name);
                cii = ciftiopen(file_path, 'wb_command');
                cii = rmfield(cii, 'cdata');
                
                
                % % Load cifti -59k
                % folder_path = '/combinelab/03_user/jungmin/02_data/03_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results/ses-movie_task-movie_run-1';
                % files = dir(fullfile(folder_path, 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'));
                % 
                % file_path = fullfile(folder_path, files.name);
                % cii = ciftiopen(file_path, 'wb_command');
                % cii = rmfield(cii, 'cdata');

                % Load correlation
                load([saveroot, '/', 'correlations', '/', '6layers_32channel', '/', '10k', '/', 'total', '/', sprintf('%s_avg_corr_svd0.90_s3.mat', feature)]);

                Y_hat = Y_hat{layer+1};

                % Update cdata and diminfo
                cii.cdata = Y_hat';
                cii.diminfo{1, 1}.length = size(Y_hat', 1);
                cii.diminfo{1, 2}.length = size(Y_hat', 2);
                cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal

                % Concatenate cdata for each feature
                if isempty(X)
                    X = cii;
                else
                    X.cdata = cat(2, X.cdata, cii.cdata);
                end  
            end

            save_file_name = ([feature,'_', 'avg','.dtseries.nii']);
            ciftisavereset(X, fullfile(saveroot,'/', 'dt_result','/', '6layers_32channel','/', '10k', '/', 'total','/',  save_file_name), 'wb_command');

            %initalize X 
            X={};
    end
end



% nanIndices = isnan(Ahat{1});
% dataWithoutNaN = Ahat{1}(~nanIndices);


%% make p_value dtseries 
clc;clear

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 


% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

X={};

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:8
        folder_name = [saveroot,'/', 'dt_result','/', '6layers_32channel','/', '10k', '/', 'total', '/']
        if not(exist(folder_name,'dir'))
            mkdir(folder_name)
        end 
            for layer=1:5

                % Load cifti -10k
                folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
                files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

                file_path = fullfile(folder_path, files.name);
                cii = ciftiopen(file_path, 'wb_command');
                cii = rmfield(cii, 'cdata');
                
                
                % % Load cifti -59k
                % folder_path = '/combinelab/03_user/jungmin/02_data/03_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results/ses-movie_task-movie_run-1';
                % files = dir(fullfile(folder_path, 'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'));
                % 
                % file_path = fullfile(folder_path, files.name);
                % cii = ciftiopen(file_path, 'wb_command');
                % cii = rmfield(cii, 'cdata');

                % Load correlation
                load([saveroot, '/', 'correlations', '/', '6layers_32channel', '/', '10k', '/', 'total', '/', sprintf('%s_avg_pval_svd0.90_s3.mat', feature)]);

                Y_hat = p_value{layer+1};

                % Update cdata and diminfo
                cii.cdata = Y_hat';
                cii.diminfo{1, 1}.length = size(Y_hat', 1);
                cii.diminfo{1, 2}.length = size(Y_hat', 2);
                cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal

                % Concatenate cdata for each feature
                if isempty(X)
                    X = cii;
                else
                    X.cdata = cat(2, X.cdata, cii.cdata);
                end  
            end

            save_file_name = ([feature,'_', 'pval_avg','.dtseries.nii']);
            ciftisavereset(X, fullfile(saveroot,'/', 'dt_result','/', '6layers_32channel','/', '10k', '/', 'total','/',  save_file_name), 'wb_command');

            %initalize X 
            X={};
    end
end


%%
% %% Step 6.1: concat Yhat for all segments + correalting with total movie fMRI
% clc; clear
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% features = {'Ahat', 'E'};
% 
% load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
% 
% for i = 4:15
%     subject = num_subjects{i}
%     disp(i)
%     disp(['Voxelwise_encoding start ', subject])    
% 
%     for feature_idx = 1:numel(features)
%         feature = features{feature_idx}
%         disp(feature);
%         for layer=0:3 
%             for seg = 1:8
%                 Y_hat = load([saveroot,'/', 'Y_hat','/','PCA', '/', '10k', '/', sprintf('seg%d',seg), '/', feature, '/', strcat(subject,'_', feature , num2str(layer), '_pred_svd0.90.mat')]);
%                 disp([saveroot,'/', 'Y_hat','/','PCA', '/', '10k', '/',sprintf('seg%d',seg), '/', feature, '/', strcat(subject,'_', feature , num2str(layer), '_pred_svd0.90.mat')])
%                 X{seg} = Y_hat.Yhat;
%             end
%             Y_pred = sprintf('%s%d_seg1_8', feature, layer)
%             concatenated.(Y_pred)= cat(1, X{:});
% 
%             %correaltions
%             %load Y 
%             sub_cii1_8 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
%             [correlations, p_values] = auto_corr(sub_cii1_8',concatenated.(Y_pred));
% 
%             correlations = correlations';
% 
%             % save correlation (feature x voxel)
%             disp('save correlations')
% 
%             name = strcat(subject,'_', feature , num2str(layer),'_corr_svd0.90_s3', '.mat')
%             save([saveroot,'/', 'correlations','/', 'PCA', '/', name], 'correlations', '-v7.3');
%         end
%     end
% end
% %% Step7: make dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% features = {'Aaht','E'};
% 
% for i = 1:15
%     subject =  num_subjects{i}
%     X={} ;
% 
%     for feature_idx = 1:numel(features)
%         feature = features{feature_idx}
%         disp(feature);
%         for layer=0:3
%             % Load cifti -10k
%             folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%             files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%             file_path = fullfile(folder_path, files.name);
%             cii = ciftiopen(file_path, 'wb_command');
%             cii = rmfield(cii, 'cdata');
% 
%             %load correaltion
%             load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', subject,'_', feature , num2str(layer),'_corr_svd0.90_s3']);
%             disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', subject,'_', feature , num2str(layer),'_corr_svd0.90_s3']);
% 
%             % Update cdata and diminfo
%             cii.cdata = correlations;
%             cii.diminfo{1, 1}.length = size(correlations, 1);
%             cii.diminfo{1, 2}.length = size(correlations, 2);
%             cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%             % Concatenate cdata for each layer
%             if isempty(X)
%                 X = cii;
%             else
%                 X.cdata = cat(2, X.cdata, cii.cdata);
%             end
%         end
% 
%         % Save as dtseries
%         save_file_name = ([subject,'_', feature, '.dtseries.nii']);
%         ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','/', save_file_name), 'wb_command');
%     end
% end
% 
% %% GUMP dir
% clc;clear
% 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_layer= 3;
% features = {'Ahat','E'};
% 
% Y_hat = cell(num_features, 1);
% 
% 
% for feature_idx = 1:numel(features)
%     feature = features{feature_idx}
%     disp(feature);
%     for layer=0:3
%         sum_values = 0;
%         for i = 1:numel(num_subjects)
%             subject = num_subjects{i};
% 
%             % Load correlation
%             load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/',   [subject, '_', feature, num2str(layer), '_corr_svd0.90_s3.mat']));
%             disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/',   [subject, '_', feature, num2str(layer), '_corr_svd0.90_s3.mat']));
% 
%             sum_values = sum_values + correlations';
%         end
%     Y_hat{layer+1} = sum_values / numel(num_subjects);
%     end
%     save(fullfile(saveroot, 'correlations', 'PCA', '10k', sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)), 'Y_hat', '-v7.3');
% end
% 
% %% Step7: make dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';
% features = {'Ahat','E'};
% 
% X={} ;
% 
% for feature_idx = 1:numel(features)
%     feature = features{feature_idx}
%     disp(feature);
%     for layer=0:3
% 
%         % Load cifti -10k
%         folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%         files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%         file_path = fullfile(folder_path, files.name);
%         cii = ciftiopen(file_path, 'wb_command');
%         cii = rmfield(cii, 'cdata');
% 
%         %load correaltion
%         load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/', sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)]);
%         disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/',sprintf('%s_avg_corrss_svd0.90_s3.mat', feature)])
%         Y_hat = Y_hat{layer+1,:};
% 
%         % Update cdata and diminfo
%         cii.cdata = Y_hat';
%         cii.diminfo{1, 1}.length = size(Y_hat', 1);
%         cii.diminfo{1, 2}.length = size(Y_hat', 2);
%         cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%         % Concatenate cdata for each feature
%         if isempty(X)
%             X = cii;
%         else
%             X.cdata = cat(2, X.cdata, cii.cdata);
%         end  
%     end
% 
%     % Save as dtseries
%     save_file_name = sprintf('%s_avg.dtseries.nii', feature);
%     ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA',  save_file_name), 'wb_command');
% 
%     % initialize X cell
%     X={} ;
% end

%% step6.2: correaltion between Y_predict and actual Y for test - single segment
% clc; clear
% disp('correaltion and p_values compute start')
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% %load Y
% load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
% 
%     disp(['correalting' , subject])
%     for feature= 0:3
% 
%         Y_pred = load([saveroot,'/', 'Y_hat','/','PCA', '/','10k', '/', 'E', '/',subject,'_', 'E' , num2str(feature), '_pred_svd0.90_s3', '.mat']);
%         Y_pred = Y_pred.Yhat;
% 
%         [correlations, p_values] = auto_corr(sub_cii{i}{8}',Y_pred);
% 
%         correlations = correlations';
% 
%         % save correlation (feature x voxel)
%         disp('save correlations')
% 
%         name = strcat(subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3', '.mat');
%         save([saveroot,'/', 'correlations','/', 'PCA', '/','10k', '/', 'E', '/', name], 'correlations', '-v7.3');
% 
%         % Find p_values < 0.05
%         p_values=p_values';
%         significant_p = p_values <= 0.05;
% 
%         % Replace numbers above 0.05 with 0
%         p_values(~significant_p) = 0;
% 
%         % Find the count of numbers in p_values
%         count = nnz(p_values);
% 
%         % save significant_p (feature x voxel)
%         disp('save p_values')
% 
%         name = strcat(subject, '_', 'E', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat');
%         save([saveroot,'/', 'p_values','/', 'PCA', '/','10k', '/', 'E', '/', name], 'p_values', '-v7.3');
%     end
% end
% % [correlations, p_values] = auto_corr(X1_56',X_pred56_sub1);
% % correlations = correlations';
% % p_values = p_values';
% 
% % % Find p_values < 0.05
% % significant_p = p_values < 0.05;
% % 
% % % Replace numbers above 0.05 with 0
% % p_values(~significant_p) = 0;
% % 
% % % Find the count of numbers in p_values
% % count = nnz(p_values);
% 
%% Step7: make dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
%     X={} ;
% 
%     for feature=0:3
% 
% %         % Load cifti -59k
% %         folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-8',subject, subject, subject);
% %         files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
% %         file_path = fullfile(folder_path, files.name);
% %         
% %         cii = ciftiopen(file_path, 'wb_command');
% %         cii = rmfield(cii, 'cdata');
% 
% 
%         % Load cifti -10k
%         folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%         files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%         file_path = fullfile(folder_path, files.name);
%         cii = ciftiopen(file_path, 'wb_command');
%         cii = rmfield(cii, 'cdata');
% 
%         %load correaltion
%         load([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/','E', '/', subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']);
%         disp([saveroot,'/', 'correlations','/','PCA', '/', '10k', '/','E', '/', subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']);
% 
%         % Update cdata and diminfo
%         cii.cdata = correlations;
%         cii.diminfo{1, 1}.length = size(correlations, 1);
%         cii.diminfo{1, 2}.length = size(correlations, 2);
%         cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%         % Concatenate cdata for each feature
%         if isempty(X)
%             X = cii;
%         else
%             X.cdata = cat(2, X.cdata, cii.cdata);
%         end
%     end
% 
%     % Save as dtseries
%     save_file_name = (['E_', subject,'.dtseries.nii']);
%     ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','10k', '/', save_file_name), 'wb_command');
% end
% 
% %% Step 8.5:avg .mat
% clc;clear
% 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_features = 3;
% 
% Y_hat = cell(num_features, 1);
% 
% for feature = 0:num_features
%     sum_values = 0;
%     for i = 1:numel(num_subjects)
%         subject = num_subjects{i};
% 
%         % Load correlation
%         load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'E', '/',  [subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']));
%         disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'E', '/',   [subject, '_', 'E', num2str(feature), '_corr_svd0.90_s3.mat']));
% 
%         sum_values = sum_values + correlations';
%     end
%     Y_hat{feature+1} = sum_values / numel(num_subjects);
% end
% save(fullfile(saveroot, 'correlations', 'PCA', '10k','E', 'E_avg_corr_svd0.90_s3.mat'), 'Y_hat', '-v7.3');
% 
% 
% 
%% Step 8.5: make avg dtseries 
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% X={} ;
% 
% for feature=0:3
% 
% %         % Load cifti -59k
% %         folder_path = sprintf('/combinelab/03_user/jungmin/01_data/03_Gump/02_FG_preprocessed/%s/%s_ciftify/ciftify/%s/MNINonLinear/Results_MNI152NLin6Asym/ses-movie_task-movie_run-8',subject, subject, subject);
% %         files = dir(fullfile(folder_path, '*s3.dtseries.nii'));
% %         file_path = fullfile(folder_path, files.name);
% %         
% %         cii = ciftiopen(file_path, 'wb_command');
% %         cii = rmfield(cii, 'cdata');
% 
% 
%     % Load cifti -10k
%     folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%     files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%     file_path = fullfile(folder_path, files.name);
%     cii = ciftiopen(file_path, 'wb_command');
%     cii = rmfield(cii, 'cdata');
% 
%     %load correaltion
%     load([saveroot,'/', 'correlations','/','PCA', '/','10k','/', 'Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
%     Y_hat = Y_hat{feature+1,:};
% 
%     % Update cdata and diminfo
%     cii.cdata = Y_hat';
%     cii.diminfo{1, 1}.length = size(Y_hat', 1);
%     cii.diminfo{1, 2}.length = size(Y_hat', 2);
%     cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%     % Concatenate cdata for each feature
%     if isempty(X)
%         X = cii;
%     else
%         X.cdata = cat(2, X.cdata, cii.cdata);
%     end  
% end
% 
% % Save as dtseries
% save_file_name = 'Ahat_avg.dtseries.nii';
% ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');
% 
%% Step 9:avg .mat -> p_value
% clc;clear
% 
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% num_features = 3;
% 
% correlation = cell(num_features, 1);
% p_value = cell(num_features, 1);
% 
% for feature = 0:num_features
%     pval_sum_values = 0;
%     corr_sum_values = 0;
%     for i = 1:numel(num_subjects)
%         subject = num_subjects{i};
% 
%         % Load correlation
%         load(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',  [subject, '_', 'Ahat', num2str(feature), '_corr_svd0.90_s3', '.mat']));
%         disp(fullfile(saveroot, 'correlations', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',   [subject, '_', 'Ahat', num2str(feature), '_corr_svd0.90_s3', '.mat']));        
% 
%         % Load p-value
%         load(fullfile(saveroot, 'p_values', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',  [subject, '_', 'Ahat', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat']));
%         disp(fullfile(saveroot, 'p_values', '/', 'PCA', '/', '10k', '/', 'Ahat', '/',   [subject, '_', 'Ahat', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat']));
% 
%         zero_indices = p_values == 0;
%         correlations(zero_indices) = 0;
%         corr_sum_values = corr_sum_values + correlations';
% 
%         pval_sum_values = pval_sum_values + p_values';
%     end
%     correlation{feature+1} = corr_sum_values / numel(num_subjects);
%     p_value{feature+1} = pval_sum_values / numel(num_subjects);
% end
% save(fullfile(saveroot, 'p_values', 'PCA', '10k','Ahat', 'Ahat_avg_corr_svd0.90_s3.mat'), 'correlation', '-v7.3');
% save(fullfile(saveroot, 'p_values', 'PCA', '10k','Ahat', 'Ahat_avg_p_values_svd0.90_s3.mat'), 'p_value', '-v7.3');
% 
%% Step9: make individual dtseries - pvalue
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/';
% num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
% 
% for i = 1:15
%     subject = num_subjects{i};
% 
%     thresholds = {0.05, 0.01, 0.001};
% 
%     for j = 1:length(thresholds)
%         threshold = thresholds{j};
%         X={} ;
% 
%         for feature=0:3
% 
%             %load p-value
%             load([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);
%             disp([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);   
% 
%             %adjust p_values
%             pValue = p_value{feature+1};
%             pValue(pValue > threshold) = 0;  
% 
%             % Load correlation
%             load([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
%             disp([saveroot, 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);  
% 
%             zero_indices = pValue == 0;
%             correlations = correlation{feature+1};
%             correlations(zero_indices) = 0;            
% 
%             % Load cifti -10k
%             folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%             files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%             file_path = fullfile(folder_path, files.name);
%             cii = ciftiopen(file_path, 'wb_command');
%             cii = rmfield(cii, 'cdata');
% 
%             % Update cdata and diminfo
%             cii.cdata = correlations';
%             cii.diminfo{1, 1}.length = size(correlations, 2); 
%             cii.diminfo{1, 2}.length = 4; % the number of layers
%             cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%             % Concatenate cdata for each feature
%             if isempty(X)
%                 X = cii;
%             else
%                 X.cdata = cat(2, X.cdata, cii.cdata);
%             end          
% 
% 
%         end
%     cii.cdata = X.cdata;
% 
%     % Save as dtseries
%     save_file_name = sprintf(['%s_Ahat_pvalues%s.dtseries.nii'], subject, num2str(threshold));
%     ciftisavereset(cii, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');   
%     end 
% end
% 
%% Step 9: make avg dtseries -> p_value
% clc;clear
% addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
% 
% %dir
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
% 
% thresholds = {0.05, 0.01, 0.001};
% 
% for j = 1:length(thresholds)
%     threshold = thresholds{j};
%     X={} ;
% 
%     for feature=0:3
% 
%         % Load cifti -10k
%         folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
%         files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
% 
%         file_path = fullfile(folder_path, files.name);
%         cii = ciftiopen(file_path, 'wb_command');
%         cii = rmfield(cii, 'cdata');
% 
%         %load p-value
%         load([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);
%         disp([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_p_values_svd0.90_s3.mat']);   
% 
%         %adjust p_values
%         pValue = p_value{feature+1};
%         pValue(pValue > threshold) = 0;  
% 
%         %load correaltion
%         load([saveroot,'/', 'p_values','/','PCA', '/', '10k','/','Ahat', '/', 'Ahat_avg_corr_svd0.90_s3.mat']);
% 
%         zero_indices = pValue == 0;
%         correlations = correlation{feature+1};
%         correlations(zero_indices) = 0;          
% 
%         % Update cdata and diminfo
%         cii.cdata = correlations';
%         cii.diminfo{1, 1}.length = size(correlations, 2); 
%         cii.diminfo{1, 2}.length = 4; % the number of layers
%         cii.diminfo{1, 1}.models(3:end) = []; % subcortex removal
% 
%         % Concatenate cdata for each feature
%         if isempty(X)
%             X = cii;
%         else
%             X.cdata = cat(2, X.cdata, cii.cdata);
%         end    
%     end
% 
%     cii.cdata = X.cdata;
%     % Save as dtseries
%     save_file_name = sprintf(['Ahat_avg_pvalues-%s.dtseries.nii'],num2str(threshold));
%     ciftisavereset(cii, fullfile(saveroot, 'dt_result', 'PCA','10k',  save_file_name), 'wb_command');
% end