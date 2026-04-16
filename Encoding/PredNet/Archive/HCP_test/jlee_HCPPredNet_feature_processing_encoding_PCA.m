
%% Step1: calculate the temporal mean and standard deviation of the feature time series of CNN units
clc;clear
% dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/02_HCP/result/h5';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

features = {'MOVIE1_Ahat', 'MOVIE2_Ahat', 'MOVIE3_Ahat', 'MOVIE4_Ahat', 'MOVIE1_E', 'MOVIE2_E', 'MOVIE3_E','MOVIE4_E'};

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
   
    name_index = strfind(feature, '_');  % Find the index of the underscore
    feature_name= feature(name_index+1:end);  % Extract the part after the underscore
    disp(feature_name);

    for i = 0:3
        % calculate the temporal mean 
        secpath = [dataroot, '/', sprintf('%s%d.h5', feature, i)];
        lay_feat = h5read(secpath,'/predimg');
        dim = size(lay_feat)
    
        lay_feat_mean = sum(lay_feat, 4) / dim(4);
        
        disp(['save average for layer', num2str(i)])
        
        h5create([saveroot, '/', 'feature_maps', '/', sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data',...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([saveroot, '/', 'feature_maps', '/', sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data', lay_feat_mean);
    
        % calculate the temporal standard deviation
        disp(['start std for layer', num2str(i)])
    
        secpath = [dataroot, '/', sprintf('%s%d.h5', feature, i)];
        lay_feat = h5read(secpath,'/predimg');
        dim = size(lay_feat);
    
        lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/', sprintf('%s%d_feature_maps_avg.h5', feature, i)],'/data');
    
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = lay_feat.^2;
        dim = size(lay_feat);
    
        lay_feat_std = zeros([dim(1:end-1),1]); 
        lay_feat_std = lay_feat_std + sum(lay_feat,length(dim));
    
        lay_feat_std = sqrt(lay_feat_std/(dim(end)-1));
        lay_feat_std(lay_feat_std==0) = 1;
        
        disp(['save average for layer', num2str(i)])
        
        h5create([saveroot, '/', 'feature_maps', '/', sprintf('%s%d_feature_maps_std.h5', feature, i)],'/data',...
            [size(lay_feat_mean)],'Datatype','single');
        h5write([saveroot,'/','feature_maps', '/',  sprintf('%s%d_feature_maps_std.h5', feature, i)],'/data',lay_feat_std);
        
        disp('start next layer!')
    end
end
disp('fin!')


%% Step2: PCA - E 
clc;clear

%dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/02_HCP/result/h5';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

% % % % % % % % % % SVD-updating algorithm % % % % % % % % %

Niter = 2; % number of iteration to compute principle component
percp = 0.90; % explain 99% of the variance of every movie segments
disp(['percp is ', num2str(percp)])

features = {'MOVIE1_E', 'MOVIE2_E', 'MOVIE3_E','MOVIE4_E'};

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
   
    name_index = strfind(feature, '_');  % Find the index of the underscore
    feature_name= feature(name_index+1:end);  % Extract the part after the underscore
    disp(feature_name);

    for i = 0:3
        disp(['layer is ', num2str(i)])
        % Step 1: Calculate mean and standard deviation
        disp('load mean and std')
        lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('%s%d_feature_maps_avg.h5', feature, i)], '/data');
        lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('%s%d_feature_maps_std.h5', feature, i)], '/data');
    
        k0 = 0;
        for iter =  1  : Niter
            disp(['iteration #: ', num2str(iter)])
            
            %load lay_feat 
            disp('load lay_feat')
            secpath = [dataroot, '/', sprintf('%s%d.h5', feature, i)];
            lay_feat = h5read(secpath,'/predimg');
        
            %normalization lay_feat
            disp('normalization begins')
            lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
            lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
            lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
    
            %reshape normalized lay_feat to features-by-time
            disp('reshape begins')
            dim = size(lay_feat);
            lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
            
            disp('amri_sig_isvd start')
    
            if iter == 1
                [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
            else
                [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
            end  
    
            disp(['Iteration ', num2str(iter), ' completed for', feature_name, num2str(i)]);
        end
        
        disp('diag(s) begins')
        
        s = diag(S);
     
        disp('start saving SVD_layer');
        % save principal components
        save([saveroot, '/', 'X', '/', 'PCA', '/', feature_name, '/', sprintf('%s%d_feature_maps_svd_0.90.mat', feature, i)], 'B', 's', '-v7.3');
        disp(['SVD_layer for', feature, ' saved']);
        
        clear B s S lay_feat
    end
end
%% Step2: PCA - Ahat
% clc;clear
% 
% %dir
% dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/02_HCP/result/h5';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';
% 
% % % % % % % % % % % SVD-updating algorithm % % % % % % % % %
% 
% Niter = 2; % number of iteration to compute principle component
% percp = 0.90; % explain 99% of the variance of every movie segments
% disp(['percp is ', num2str(percp)])
% 
% features = {'MOVIE1_Ahat', 'MOVIE2_Ahat', 'MOVIE3_Ahat', 'MOVIE4_Ahat'};
% 
% for feature_idx = 1:numel(features)
%     feature = features{feature_idx};
%     disp(feature);
% 
%     name_index = strfind(feature, '_');  % Find the index of the underscore
%     feature_name= feature(name_index+1:end);  % Extract the part after the underscore
%     disp(feature_name);
% 
%     for i = 2:3
%         disp(['layer is ', num2str(i)])
%         % Step 1: Calculate mean and standard deviation
%         disp('load mean and std')
%         lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('%s%d_feature_maps_avg.h5', feature, i)], '/data');
%         lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('%s%d_feature_maps_avg.h5', feature, i)], '/data');
% 
%         k0 = 0;
%         for iter =  1  : Niter
%             disp(['iteration #: ', num2str(iter)])
% 
%             %load lay_feat 
%             disp('load lay_feat')
%             secpath = [dataroot, '/', sprintf('%s%d.h5', feature, i)];
%             lay_feat = h5read(secpath,'/predimg');
% 
%             %normalization lay_feat
%             disp('normalization begins')
%             lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
%             lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
%             lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
% 
%             %reshape normalized lay_feat to features-by-time
%             disp('reshape begins')
%             dim = size(lay_feat);
%             lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
% 
%             disp('amri_sig_isvd start')
% 
%             if iter == 1
%                 [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp);
%             else
%                 [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', {B,S});
%             end  
% 
%             disp(['Iteration ', num2str(iter), ' completed for ', feature_name, num2str(i)]);
%         end
% 
%         disp('diag(s) begins')
% 
%         s = diag(S);
% 
%         disp('start saving SVD_layer');
%         % save principal components
%         save([saveroot, '/', 'X', '/', feature_name, '/', sprintf('%s%d_feature_maps_svd_0.90.mat', feature, i)], 'B', 's', '-v7.3');
%         disp(['SVD_layer for Ahat', num2str(i), ' saved']);
%         clear B s S 
%     end
% end
%% Step3: Processed the hrf
clc; clear 

% dir
dataroot = '/combinelab/03_user/jungmin/01_project/01_PredNet/jmPNET/data/02_HCP/result/h5';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

% The sampling rate should be equal to the sampling rate of CNN feature
% maps. If the CNN extracts the feature maps from movie frames with 30
% frames/second, then srate = 30. It's better to set srate as even number
% for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 24; %original code is at TR=2, [if TR=1 && srate is 25] then make it into 12.5 

% Here is an example of using pre-defined hemodynamic response function
% (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32];
hrf = spm_hrf(1/srate,p);
hrf = hrf(:);
% figure; plot(0:1/srate:p(7),hrf);
tic
features = {'MOVIE1_Ahat', 'MOVIE2_Ahat', 'MOVIE3_Ahat', 'MOVIE4_Ahat', 'MOVIE1_E', 'MOVIE2_E', 'MOVIE3_E','MOVIE4_E'};

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature);
   
    name_index = strfind(feature, '_');  % Find the index of the underscore
    feature_name= feature(name_index+1:end);  % Extract the part after the underscore
    disp(feature_name);
    for i=0:3
        disp(num2str(i))
        % Dimension reduction for features
        disp('load mean and std')
        lay_feat_mean = h5read([saveroot, '/', 'feature_maps', '/',  sprintf('%s%d_feature_maps_avg.h5', feature, i)], '/data');
        lay_feat_std = h5read([saveroot, '/','feature_maps', '/', sprintf('%s%d_feature_maps_std.h5', feature, i)], '/data');
        load([saveroot,'/', 'X', '/','PCA', '/', feature_name,'/', sprintf('%s%d_feature_maps_svd_0.90.mat', feature, i)], 'B');
        
        disp([saveroot,'/', 'X', '/','PCA', '/', feature_name,'/', sprintf('%s%d_feature_maps_svd_0.90.mat', feature, i)])

        %load lay_feat 
        disp('load lay_feat')
        secpath = [dataroot, '/', sprintf('%s%d.h5', feature, i)];
        lay_feat = h5read(secpath,'/predimg');
        
        disp('compute lay_feat regulaization')
        lay_feat = bsxfun(@minus, lay_feat, lay_feat_mean);
        lay_feat = bsxfun(@rdivide, lay_feat, lay_feat_std);
        lay_feat(isnan(lay_feat)) = 0; % assign 0 to nan values
    
        disp('reshape lay_feat')
        dim = size(lay_feat);
        lay_feat = reshape(lay_feat, prod(dim(1:end-1)),dim(end));
    
        disp('compute X with svd applied')
        X = lay_feat'*B/sqrt(size(B,1)); % Y: #time-by-#components
    
        disp('compute ts')
        ts = conv2(hrf,X); % convolude with hrf
        ts = ts(4*srate+1:4*srate+size(X,1),:);
        ts = ts(srate:1*srate:end,:); % downsampling to match fMRI  %OG TR at 2 => ts = ts(srate+1:2*srate:end,:); 
    
        disp('start saving')
        
        filename = sprintf('%s%d_feature_maps_svd_0.90_hrf.mat', feature, i);
        save([saveroot,'/', 'X', '/', 'PCA', '/', feature_name, '/',  filename], 'ts', '-v7.3');
        
        disp([saveroot,'/', 'X', '/', 'PCA', '/', feature_name, '/',  filename])
        disp(['SVD_layer for ',feature_name, num2str(i), ' saved']);
    
        
        clear ts lay_feat
    end
end




%% Step5.1: Train Encoding with HCP 2mm smoothing - 10k
% Load Y
clc;clear 
addpath(genpath('/local_raid1/01_software/toolboxes/cifti-matlab'));
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
taskIDs = {'tfMRI_MOVIE1_7T_AP', 'tfMRI_MOVIE2_7T_PA', 'tfMRI_MOVIE3_7T_PA', 'tfMRI_MOVIE4_7T_AP'};
sub_cii = cell(numel(subjectIDs), 1);
existing_subjects = {};  % To store the subjects that are found
subcortex_no = 1:18722;

for subject_idx = 1:numel(subjectIDs)
    subject = subjectIDs(subject_idx);
    subject_cii = cell(1, numel(taskIDs));
    subject_exists = true;  % Flag to indicate if subject exists
         
    disp(['start next sub ', num2str(subject)])

    for task = 1:numel(taskIDs)
        task_idx = taskIDs{task};
        disp(task_idx)
        folder_path = sprintf('/combinelab/02_data/01_HCP/7T_MOVIE_2mm/%d/MNINonLinear/Results/%s', subject, task_idx);
        
        files = fullfile(folder_path, sprintf('%s_Atlas_MSMAll.10k.dtseries.nii', task_idx));

        cii = ciftiopen(files, 'wb_command');
        cii_10k = cii.cdata(subcortex_no, :);

        subject_cii{task} = cii_10k; 
        
        % Check if subject data is empty
        if isempty(subject_cii{task})
            subject_exists = false;
            break;  % Exit the loop if subject data is empty for any run
        end
    end
    
    
    % Store subject data if it exists for all runs
    if subject_exists
        existing_subjects = [existing_subjects, subject];  % Append to existing_subjects
        sub_cii{subject_idx} = subject_cii;
    end
end

% Save existing subjects
save_file = [saveroot, '/', 'Y', '/', 'fMRI_10k_cortex'];
save(save_file, 'sub_cii', 'existing_subjects', '-v7.3');

%% Step5.2:X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

features = {'Ahat','E'};


for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:3
        for seg = 1:4
            filename = sprintf('%s/X/PCA/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', saveroot, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.segments;
            eval([name, '= concatenated.(name);']);
%             X_total{end+1, 1} = name;  % Store the name
%             X_total{end, 2} = loaded_data.segments;  % Store the segments
        end
    end

end
 
%Voxelwise_encoding
lambda = [0.1:0.2:0.9];
nfold = 9; %from paper 

%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);
subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
features = {'Ahat' , 'E'};


for i = 1:15
    subject = subjectIDs(i);
    disp(['Voxelwise_encoding start sub ', num2str(subject)])
    
    for feature_idx = 1:numel(features)
        feature = features{feature_idx};
        disp(feature)

        for seg = 1:4
            disp(['Segment: ', num2str(seg)]);
           for layer = 0:3
                disp(['Layer: ', num2str(layer)]);
                
                % Y calculate 
                switch seg
                    case 1
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{2}, sub_cii{i}{3}, sub_cii{i}{4});
                    case 2
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{3}, sub_cii{i}{4});
                    case 3
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{4});
                    case 4
                        disp(['seg is ', num2str(seg)])
                        Y = cat(2, sub_cii{i}{1}, sub_cii{i}{2}, sub_cii{i}{3});
                end
                
                % X calculate
                X_tot = {};
                for s = 1:4
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
                folder_name = [saveroot,'/', '/lambda/', 'W','/','PCA','/',sprintf('seg%d',seg)]
                if not(exist(folder_name,'dir'))
                    mkdir(folder_name)
                end
                
                name = [num2str(subject), '_', feature_name, num2str(feature), '_W_svd0.9_s3.mat'];
                save([saveroot,'/lambda/', 'W','/','PCA','/',sprintf('seg%d',seg),'/', name], 'W', '-v7.3');
            end
        end
    end
end

%% Step6.1: Load test Y and X

% X model featuremaps 
clc; clear
disp('Load model featuremaps')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

features = {'Ahat','E'};

for feature_idx = 1:numel(features)
    feature = features{feature_idx};
    disp(feature); 
    for layer = 0:3
        for seg = 1:4
            filename = sprintf('%s/X/PCA/%s%d_feature_maps_svd_0.90_hrf_seg%d.mat', saveroot, feature, layer, seg);
            loaded_data = load(filename);
            name = sprintf('%s%d_seg%d', feature, layer, seg)
            concatenated.(name) = loaded_data.segments;
            eval([name, '= concatenated.(name);']);
        end
    end

end
disp('make Y_hat')
subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];

for i = 1:15
    subject = subjectIDs(i)
    disp(['Voxelwise_encoding start sub ', num2str(subject)])    
    
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        
        for seg=1:4
            folder_name = [saveroot,'/', 'Y_hat','/','PCA', '/', sprintf('seg%d',seg)]
            if not(exist(folder_name,'dir'))
                mkdir(folder_name)
            end
            for layer=0:3

                % X calculate
                X = {};
                for s = 1:4
                    if s == seg
                        disp(['s is ', num2str(s)])
                        disp([sprintf('%s%d_seg%d', feature, layer, s)])
                        X = concatenated.(sprintf('%s%d_seg%d', feature, layer, s));
                    end
                end

                W = load([saveroot,'/', 'W', '/', 'PCA',  '/', sprintf('seg%d',seg), '/',num2str(subject), '_', feature, num2str(layer), '_W_svd0.9.mat']);
                name = ['Y_pred4_sub',num2str(subject),'_',feature, num2str(layer)] 
                eval([name, '= X * W.W;']);
                
                %save Y_hat
                Yhat = eval(name);
                name2 = strcat(num2str(subject),'_', feature , num2str(layer), '_pred_svd0.90.mat');
               
                save([saveroot,'/', 'Y_hat','/','PCA', '/', sprintf('seg%d',seg),  '/',  name2], 'Yhat', '-v7.3');
            end
        end
    end
end

%% step6.2: correaltion between Y_predict and actual Y for test
clc; clear
disp('correaltion and p_values compute start')

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

features = {'Ahat','E'};


%load Y
load([saveroot, '/' , 'Y', '/', 'fMRI_10k_cortex.mat']);

subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];

for i = 1:15
    subject = subjectIDs{i};
    
    disp(['correalting' , subject])
    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        
        for seg=1:4
            folder_name = [saveroot,'/', 'Y_hat','/','PCA', '/', sprintf('seg%d',seg)]
            if not(exist(folder_name,'dir'))
                mkdir(folder_name)
            end
            for layer= 0:3
                
                Y_pred = load([saveroot,'/', 'Y_hat','/','PCA', '/',sprintf('seg%d', seg),'/',  subject,'_', 'Ahat' , num2str(feature), '_pred_svd0.90_s3', '.mat']);
                Y_pred = Y_pred.Yhat;
                
                [correlations, p_values] = auto_corr(sub_cii{i}{8}',Y_pred);
                
                correlations = correlations';
                
                % save correlation (feature x voxel)
                disp('save correlations')
                
                name = strcat(subject, '_', feature, num2str(layer), '_corr_svd0.90_s3', '.mat');
                save([saveroot,'/', 'correlations','/','PCA', '/',sprintf('seg%d', seg),'/', name], 'correlations', '-v7.3');
                
                %             % Find p_values < 0.05
                %             p_values=p_values';
                %             significant_p = p_values < 0.05;
                %
                %             % Replace numbers above 0.05 with 0
                %             p_values(~significant_p) = 0;
                %
                %             % Find the count of numbers in p_values
                %             count = nnz(p_values);
                %
                %             % save significant_p (feature x voxel)
                %             disp('save p_values')
                %
                %             name = strcat(subject, '_', 'Ahat', num2str(feature), '_svd0.90_p_values_0.05_s3', '.mat');
                %             save([saveroot,'/', 'p_values','/',sprintf('seg%d', seg),'/' name], 'p_values', '-v7.3');
            end
        end
    end
end
% [correlations, p_values] = auto_corr(X1_56',X_pred56_sub1);
% correlations = correlations';
% p_values = p_values';

% % Find p_values < 0.05
% significant_p = p_values < 0.05;
% 
% % Replace numbers above 0.05 with 0
% p_values(~significant_p) = 0;
% 
% % Find the count of numbers in p_values
% count = nnz(p_values);

%% Step7: make dtseries 
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

%dir
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';
subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
features = {'Ahat'};

for i = 1:15
    subject = subjectIDs(i)
    X={} ;

    for feature_idx = 1:numel(features)
        feature = features{feature_idx}
        disp(feature);
        for layer=0:3
            % Load cifti -10k
            folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
            files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
    
            file_path = fullfile(folder_path, files.name);
            cii = ciftiopen(file_path, 'wb_command');
            cii = rmfield(cii, 'cdata');
    
            %load correaltion
            load([saveroot,'/', 'correlations','/','PCA', '/', num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3']);
            disp([saveroot,'/', 'correlations','/','PCA', '/', num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90_s3']);
            
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
        ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','/', save_file_name), 'wb_command');
    end
end

%% avg .mat
clc;clear

%Titanic trained
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

%500SUM trained
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_HCP/result/h5';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

subjectIDs = [100610, 116726, 135124, 156334, 169343, 178243, 191841, 201515, 249947, 380036, 463040, 725751, 818859, 901139, 102311, 118225, 137128, 157336, 169444, 178647, 192439, 203418, 251833, 381038, 467351, 601127, 732243, 825048, 901442, 995174, 102816, 125525, 140117, 158035, 169747, 180533, 192641, 204521, 257845, 385046, 617748, 826353, 905147, 104416, 126426, 144226, 158136, 171633, 181232, 193845, 205220, 263436, 389357, 525541, 627549, 751550, 833249, 910241, 105923, 145834, 159239, 172130, 195041, 209228, 283543, 393247, 638049, 757764, 859671, 926862, 108323, 128935, 146129, 162935, 173334, 182436, 196144, 212419, 318637, 395756, 541943, 644246, 765864, 861456, 927359, 109123, 130114, 146432, 164131, 175237, 182739, 197348, 214019, 320826, 397760, 547046, 654552, 770352, 871762, 942658, 111312, 130518, 146735, 164636, 176542, 185442, 198653, 214524, 330324, 401422, 671855, 771354, 872764, 943862, 111514, 131722, 146937, 165436, 177140, 186949, 199655, 221319, 346137, 406836, 562345, 680957, 782561, 878776, 951457, 114823, 132118, 148133, 167036, 177645, 187345, 200210, 233326, 352738, 412528, 572045, 690152, 783462, 878877, 958976, 115017, 134627, 150423, 167440, 177746, 191033, 200311, 239136, 360030, 429040, 573249, 706040, 789373, 898176, 966975, 115825, 134829, 155938, 169040, 178142, 191336, 200614, 246133, 365343, 436845, 581450, 724446, 814649, 899885, 971160];
num_layer = 3;
features = {'Ahat', 'E'}

Y_hat = {};


for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:4
        %         folder_name = [saveroot,'/', 'dt_result','/', 'PCA', '/', sprintf('seg%d',seg),'/']
        %         if not(exist(folder_name,'dir'))
        %             mkdir(folder_name)
        %         end
        for layer= 0:3
            sum_values = 0;
            for i = 1:15
                subject = subjectIDs(i)

                % Load correlation
                load([saveroot,'/', 'correlations','/', 'PCA', '/', sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90.mat')]);
                disp([sprintf('seg%d',seg), '/', strcat(num2str(subject),'_', feature , num2str(layer),'_corr_svd0.90.mat')]);

                sum_values = sum_values + correlations';
            end
            Y_hat{layer+1} = sum_values / 15;
        end
        save([saveroot, '/', 'correlations', '/', 'PCA', '/', sprintf('seg%d',seg),  '/', sprintf('%s_avg_corr_svd0.90_s3.mat', feature)], 'Y_hat', '-v7.3');
        Y_hat = {};
    end
end

%% make avg dtseries 
clc;clear
addpath(genpath('/local_raid1/01_software/HCPpipelines/global/matlab/cifti-matlab'));

%dir
%Titanic trained
dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/03_Titanic/result/HCP/4layers';
saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/HCP_test';

%500SUM trained
% dataroot = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_HCP/result/h5';
% saveroot = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/500SUM_CIT_trained/HCP_test';

features = {'Ahat', 'E'}
X={};

for feature_idx = 1:numel(features)
    feature = features{feature_idx}
    disp(feature)
    for seg = 1:4
%         folder_name = [saveroot,'/', 'dt_result','/', 'PCA', '/', sprintf('seg%d',seg),'/', feature]
%         if not(exist(folder_name,'dir'))
%             mkdir(folder_name)
%         end 
            for layer=0:3
            
                % Load cifti -10k
                folder_path = '/combinelab/02_data/01_HCP/7T_MOVIE_2mm/100610/MNINonLinear/Results/tfMRI_MOVIE1_7T_AP/';
                files = dir(fullfile(folder_path, 'tfMRI_MOVIE1_7T_AP_Atlas_MSMAll.10k.dtseries.nii'));
            
                file_path = fullfile(folder_path, files.name);
                cii = ciftiopen(file_path, 'wb_command');
                cii = rmfield(cii, 'cdata');
            
                % Load correlation
                load([saveroot, '/', 'correlations', '/', 'PCA', '/', sprintf('seg%d',seg),  '/', sprintf('%s_avg_corr_svd0.90_s3.mat', feature)]);
         
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
            ciftisavereset(X, fullfile(saveroot, 'dt_result', 'PCA','/', sprintf('seg%d',seg), '/',  save_file_name), 'wb_command');

            %initalize X 
            X={};
    end
end
