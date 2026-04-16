%% Step3: Processed the hrf 
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/4layers_32channel/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

srate = 25; %original code is at TR=2, [if TR=1 && srate is 25] then make it into 12.5 

% p  = [5, 16, 1, 1, 6, 0, 32];
% hrf = spm_hrf(1/srate,p);
% hrf = hrf(:);

Prediction_features = {'run1_Ahat', 'run2_Ahat', 'run3_Ahat', 'run4_Ahat', 'run5_Ahat', 'run6_Ahat', 'run7_Ahat', 'run8_Ahat'}
Error_features={'run1_E', 'run2_E', 'run3_E', 'run4_E', 'run5_E', 'run6_E', 'run7_E', 'run8_E'};

svd_folder = [saveroot,'/', 'X','/', '4layers_32channel', '/'];

%Prediction
disp('START PREDICTION FEATURE');
for lay=1:3
    disp(num2str(lay))

    lay_feat_mean = h5read([saveroot, '/', 'feature_maps','/', '4layers_32channel', '/',sprintf('Ahat%d_feature_maps_avg.h5', lay)], '/data');
    disp(sprintf('Ahat%d_feature_maps_avg.h5', lay))

    lay_feat_std = h5read([saveroot, '/', 'feature_maps','/', '4layers_32channel', '/', sprintf('Ahat%d_feature_maps_std.h5', lay)], '/data');
    disp(sprintf('Ahat%d_feature_maps_std.h5', lay))

    load([svd_folder, sprintf('Ahat%d_feature_maps_svd_0.90.mat', lay)], 'B');
    disp(sprintf('Ahat%d_feature_maps_svd_0.90.mat', lay))


    for seg = 1 : 8
        feature = Prediction_features{seg}
        
        feature_name = feature(6:end);
        disp(feature_name)
        
        seg_name = feature(1:4);
        disp(seg_name)

        %load lay_feat
        disp('load lay_feat')
        secpath = [dataroot, '/', 'h5', '/', seg_name, '/', sprintf('%s%d.h5', feature_name, lay)]
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
        % ts = conv2(hrf,X); % convolude with hrf
        % ts = ts(4*srate+1:4*srate+size(X,1),:);
        ts = X(srate+1:2*srate:end,:); % downsampling to match fMRI

        disp('start saving')

        filename = [sprintf('%s%d_feature_maps_svd_0.90_no_hrf_seg%s.mat', feature_name,lay, num2str(seg))]
        save([svd_folder, filename], 'ts', '-v7.3');

        clear ts lay_feat
    end
end

%Error
disp('START ERROR FEATURE');

for lay=0:3
    disp(num2str(lay))

    lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/','4layers_32channel', '/',  sprintf('E%d_feature_maps_avg.h5',lay)], '/data');
    disp(sprintf('E%d_feature_maps_avg.h5', lay))

    lay_feat_std = h5read([saveroot, '/','feature_maps', '/','4layers_32channel', '/', sprintf('E%d_feature_maps_std.h5',  lay)], '/data');
    disp(sprintf('E%d_feature_maps_std.h5', lay))

    load([svd_folder, sprintf('E%d_feature_maps_svd_0.90.mat', lay) ], 'B');
    disp(sprintf('E%d_feature_maps_svd_0.90.mat', lay))

    for seg = 1 : 8
        feature = Error_features{seg}
        
        feature_name = feature(6:end);
        disp(feature_name)
        
        seg_name = feature(1:4);
        disp(seg_name)

        %load lay_feat
        disp('load lay_feat')
        secpath = [dataroot, '/', 'h5', '/', seg_name, '/', sprintf('%s%d.h5', feature_name, lay)];
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
        % ts = conv2(hrf,X); % convolude with hrf
        % ts = ts(4*srate+1:4*srate+size(X,1),:);
        ts = X(srate+1:2*srate:end,:); % downsampling to match fMRI

        disp('start saving')

        filename = [sprintf('%s%d_feature_maps_svd_0.90_no_hrf_seg%s.mat', feature_name, lay, num2str(seg))];
        save([svd_folder, filename], 'ts', '-v7.3');

        clear ts lay_feat
    end
end

%% Step4:W
clc; clear
disp('Load model featuremaps')

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/4layers_32channel/';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';

features = { 'Ahat', 'E'};
svd_folder = [saveroot,'/', 'X','/', '4layers_32channel', '/'];

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:3
        for seg = 1:8
            filename = sprintf('%s/%s%d_feature_maps_svd_0.90_no_hrf_seg%d.mat', svd_folder, feature, layer, seg);
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
load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 14:15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:8
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:3
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
                folder_name = [saveroot,'/', 'W','/','4layers_32channel','/', '10k', '/', 'no_hrf', '/', sprintf('seg%d',seg), '/']
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end
                
                name = [num2str(subject), '_', feature, num2str(layer), '_W_no_hrf_svd0.9_s3.mat']
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
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

features = {'Ahat','E'};

svd_folder = [saveroot,'/', 'X','/', '4layers_32channel', '/'];

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:3
        for seg = 1:8
            filename = sprintf('%s/%s%d_feature_maps_svd_0.90_no_hrf_seg%d.mat', svd_folder, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.ts;
            % eval([name, '= concatenated.(name);']);
        end
    end
end

%load Y
load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};

for i = 15
    subject = num_subjects{i};
    disp(['Voxelwise_encoding start sub', num2str(i)])
    
   for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:8
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:3
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

                W = load([saveroot,'/', 'W','/', '4layers_32channel', '/', '10k', '/', 'no_hrf', '/', sprintf('seg%d',seg), '/', num2str(subject), '_', feature, num2str(layer), '_W_no_hrf_svd0.9_s3.mat']);
                name = ['Y_pred_', feature, num2str(layer)] 
                eval([name, '= X*W.W;']);
                
                folder_name = [saveroot,'/', 'Y_hat','/','4layers_32channel', '/', '10k','/', 'no_hrf',  '/',sprintf('seg%d',seg), '/']
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end
                
                %save Y_hat
                Yhat = eval(name);
                name2 = strcat(num2str(subject),'_', feature , num2str(layer), '_pred_no_hrf_svd0.9_s3.mat');
               
                save([folder_name,  name2], 'Yhat', '-v7.3');
            end
        end
    end
end


%% step6: correaltion between Y_predict and actual Y for test - single segment
clc; clear
disp('correaltion and p_values compute start')

% dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);


for i = 15
    subject = num_subjects{i};

    disp(['correalting' , subject])

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature)

        for layer= 0:3
            for seg = 1:8
                folder_name = [saveroot,'/', 'correlations','/',  '4layers_32channel', '/', '10k', '/', 'no_hrf', '/', sprintf('seg%d',seg)]
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end

                Y_pred = load([saveroot,'/', 'Y_hat','/','4layers_32channel','/','10k', '/', 'no_hrf', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pred_no_hrf_svd0.9_s3', '.mat')]);
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

                name = strcat(num2str(subject),'_', feature , num2str(layer), '_corr_svd_no_hrf_0.9_S3.mat')
                save([saveroot,'/', 'correlations','/', '4layers_32channel','/','10k', '/','no_hrf', '/', sprintf('seg%d',seg), '/', name], 'correlations', '-v7.3');

            end
        end
    end
end

%% Step 6: concat Yhat for all segments + correalting with total movie fMRI
clc; clear

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

folder_name = [saveroot,'/', 'correlations','/', '4layers_32channel','/', 'total', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

for i = 1:15
    subject = num_subjects{i};
    disp(i)
    disp(['Voxelwise_encoding start sub ', num2str(subject)])    

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer= 0:3
            for seg = 1:8
                Y_hat = load([saveroot,'/', 'Y_hat','/','4layers_32channel','/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pred_svd0.9_s3', '.mat')]);
                disp([saveroot,'/', 'Y_hat','/','4layers_32channel','/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer), '_pred_svd0.9_S3.mat')])

                %zscore
                X{seg} = zscore(Y_hat.Yhat);

            end
            Y_pred = sprintf('%s%d_seg1_8', feature, layer)

            %concatenated zscored Y_hat for seg1-4
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
        end
    end
end


%% total mat
% clc; clear

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

load([fmriroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

folder_name = [saveroot,'/', 'correlations','/', '4layers_32channel','/','10k', '/', 'no_hrf', '/', 'total', '/']
if not(exist(folder_name,'dir'))
    mkdir(folder_name)
end

for i = 1:15
    subject = num_subjects{i};
    disp(i)
    disp(['Voxelwise_encoding start sub ', num2str(subject)])    

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer= 0:3
            for seg = 1:8
                Y_hat = load([saveroot,'/', 'Y_hat','/','4layers_32channel','/',  '10k', '/', 'no_hrf', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_pred_no_hrf_svd0.9_s3', '.mat')]);

                %zscore
                X{seg} = zscore(Y_hat.Yhat);

            end
            Y_pred = sprintf('%s%d_seg1_8', feature, layer)

            %concatenated zscored Y_hat for seg1-4
            concatenated.(Y_pred)= cat(1, X{:});

            %correaltions
            %load Y 
            sub_cii1_8 = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4}, sub_cii{i}{5}, sub_cii{i}{6}, sub_cii{i}{7}, sub_cii{i}{8});
            [correlations, p_values] = auto_corr(sub_cii1_8',concatenated.(Y_pred));

            correlations = correlations';

            % save correlation (feature x voxel)
            disp('save correlations')

            name = strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd_no_hrf_0.90_s3', '.mat')
            save([folder_name, name], 'correlations', '-v7.3');
        end
    end
end



%% avg .mat
clc;clear

addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab')); 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/GUMP/6layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test';
fmriroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/GUMP_test';

num_subjects = {'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'};
features = {'Ahat', 'E'};

Y_hat = {};

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:8
        folder_name = [saveroot,'/', 'dt_result','/', '4layers_32channel','/', '10k', '/', 'no_hrf','/', 'total', '/']
        if not(exist(folder_name,'dir'))
            mkdir(folder_name)
        end 
        for layer= 0:3
            sum_values = 0;
            for i = 1:15
                subject = num_subjects{i}

                % Load correlation
                load([saveroot,'/', 'correlations','/','4layers_32channel','/', '10k', '/', 'no_hrf','/', 'total', '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd_no_hrf_0.90_s3.mat')]);
                disp([sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3.mat')]);

                sum_values = sum_values + correlations';
            end

        Y_hat{layer+1} = sum_values / 15;
        end
        save([saveroot, '/', 'correlations', '/','4layers_32channel','/',  '10k','/', 'no_hrf','/',  'total', '/', sprintf('%s_avg_corr_svd_no_hrf_0.90_s3.mat', feature)], 'Y_hat', '-v7.3');

        %initalize 
        Y_hat = {};
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
        folder_name = [saveroot,'/', 'dt_result','/', '4layers_32channel','/',  '10k', '/',  'no_hrf','/',  'total', '/']
        if not(exist(folder_name,'dir'))
            mkdir(folder_name)
        end 
            for layer=0:3

                % Load cifti -10k
                folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
                files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));

                file_path = fullfile(folder_path, files.name);
                cii = ciftiopen(file_path, 'wb_command');
                cii = rmfield(cii, 'cdata');

                % Load correlation
                load([saveroot, '/', 'correlations', '/', '4layers_32channel','/',  '10k', '/',  'no_hrf', '/',  'total', '/', sprintf('%s_avg_corr_svd_no_hrf_0.90_s3.mat', feature)]);

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
            ciftisavereset(X, fullfile(saveroot,'/', 'dt_result','/', '4layers_32channel','/',  '10k', '/', 'no_hrf', '/',  'total', '/',  save_file_name), 'wb_command');

            %initalize X 
            X={};
    end
end
